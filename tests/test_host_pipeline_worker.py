from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch
import torch
from dllm.cpu_hybrid_stage import CpuHybridStage
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.host_completion import HostCompletion
from dllm.host_pipeline_worker import HostPipelineCoordinator,HostPipelineWorker
from dllm.host_topology import HostTopology
from dllm.host_tensor_channel import HostTensorChannel
from dllm.planned_host_stage_channel import PlannedHostStageChannel
from dllm.stage_protocol import StageState
from tests.test_cpu_hybrid_stage import model,decoded
from tests.test_cpu_pipeline_generation import OPTIONS
from tests import test_host_tls


class HostPipelineWorkerTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def session(self,mode='none',count=3,*,ready=True):
        source=model(mode);ranges=[(0,1),(1,2),(2,4)] if count==3 else [(i,i+1) for i in range(4)]
        plan=HostTopology(source.config,ranges,b'p'*32,quantization=mode,context=16,slots=2,
                          chunk_size=4,frame_bytes=4096,mapped_vocab_size=60,excluded_ids=[59])
        stages=[CpuHybridStage(source,a,b,max_batch=2,**OPTIONS) for a,b in ranges]
        for stage in stages:self.addCleanup(stage.close)
        links=[];workers=[]
        for index in range(1,count):
            fixture=test_host_tls.HostTlsTests();self.addCleanup(fixture.doCleanups)
            pair=[]
            for rank,sock in enumerate(fixture.pair()):
                transport=HostTensorChannel(sock,plan.stage_identity(index),rank,max_frame_bytes=4096,timeout=2)
                self.addCleanup(transport.close)
                channel=PlannedHostStageChannel(transport,StageState(7,16,2),plan,index)
                pair.append(channel)
            links.append(pair[0]);workers.append(HostPipelineWorker(stages[index],pair[1]))
        coordinator=HostPipelineCoordinator(stages[0],plan,links);owners=[coordinator,*workers]
        for owner in owners:self.addCleanup(owner.close)
        if ready:
            with ThreadPoolExecutor(count) as pool:
                futures=[pool.submit(w.handshake) for w in owners]
                for future in futures:future.result(5)
        return source,owners

    def command(self,owners,call,*,fails=False):
        with ThreadPoolExecutor(len(owners)) as pool:
            futures=[pool.submit(call),*[pool.submit(w.step) for w in owners[1:]]]
            results=[]
            for future in futures:
                if fails:
                    with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):future.result(5)
                else:results.append(future.result(5))
            return results[0] if results else None

    def pump(self,owners):
        errors=[];threads=[]
        def serve(worker):
            try:
                while not worker.closed:worker.step()
            except (RuntimeError,ValueError,EOFError,OSError) as error:errors.append(error)
            finally:worker.close()
        for worker in owners[1:]:
            thread=threading.Thread(target=serve,args=(worker,),daemon=True);thread.start();threads.append(thread)
        def close():
            for owner in owners:owner.close()
            for thread in threads:thread.join(3);self.assertFalse(thread.is_alive())
        self.addCleanup(close)
        return errors,threads

    def test_three_and_four_stages_match_expanded_logits_and_cache_state(self):
        for mode in ('none','q4_0','q8_0'):
            for count in (3,4):
                source,owners=self.session(mode,count);left=owners[0]
                full=CpuHybridStage(decoded(source),0,4,max_batch=2,**OPTIONS);self.addCleanup(full.close)
                captured=[];output=owners[-1].stage.output_logits
                def capture(values):
                    result=output(values);captured.append(result.clone());return result
                def execute(ids,positions,slots,prefill=True,start=0):
                    hidden=full.embedding(torch.tensor(ids))
                    result=full.forward_layers(hidden,torch.tensor(positions),torch.tensor(slots),prefill,start)
                    logits=full.output_logits(result[:,-1]);scores=logits[:,:60].clone();scores[:,59]=-float('inf')
                    return logits,scores.argmax(-1).tolist()
                with patch.object(owners[-1].stage,'output_logits',side_effect=capture):
                    for ids,start in (([1,2],0),([3,4],2)):
                        logits,expected=execute([ids],[[start,start+1]],[0],True,start)
                        self.assertEqual(self.command(owners,lambda:left.prefill_chunk(ids,0,start)),expected[0])
                        torch.testing.assert_close(captured[-1],logits,rtol=2e-4,atol=2e-6)
                    logits,expected=execute([[5,6]],[[0,1]],[1])
                    self.assertEqual(self.command(owners,lambda:left.prefill([5,6],1)),expected[0])
                    for ids,slots,positions in (([7,8],[1,0],[[2],[4]]),([9],[0],[[5]])):
                        logits,expected=execute([[v] for v in ids],positions,slots,False)
                        self.assertEqual(self.command(owners,lambda:left.decode(ids,slots)),expected)
                        torch.testing.assert_close(captured[-1],logits,rtol=2e-4,atol=2e-6)
                for owner in owners:
                    self.assertEqual(owner.stage.lengths,[6,3])
                    for name in ('kv','conv','state'):
                        for layer,value in getattr(owner.stage,name).items():
                            torch.testing.assert_close(value,getattr(full,name)[layer],rtol=2e-4,atol=2e-6)
                self.command(owners,lambda:left.release([0,1]))
                self.command(owners,lambda:left.prefill_batch([[1],[2,3]],[1,0]))
                self.assertTrue(all(w.stage.lengths==[2,1] for w in owners))
                self.command(owners,left.ping);self.command(owners,left.stop)
                self.assertTrue(all(w.closed and w.stage.closed and not w.stage.w for w in owners))

    def test_ragged_completion_release_reuse_and_eos(self):
        for mode in ('none','q4_0','q8_0'):
            source,owners=self.session(mode,4);errors,threads=self.pump(owners);left=owners[0]
            runner=HostCompletion(left);control=CpuPipelineReferenceGenerator(source,[2**22],vocab_size=60,excluded_ids=[59],**OPTIONS)
            self.addCleanup(control.cleanup)
            rows=[[1,2,3,4],[5]]
            self.assertEqual(list(runner.generate_batch(rows,max_new_tokens=[1,4],chunk_size=2)),
                             list(control.generate_batch(rows,max_new_tokens=[1,4],chunk_size=2)))
            self.assertTrue(all(w.stage.lengths==[0,0] for w in owners))
            probe=list(runner.generate([1,2],max_new_tokens=1));runner.eos_ids=frozenset([probe[0]['token_id']])
            with patch.object(left,'decode',side_effect=AssertionError('decode after EOS')):
                self.assertEqual(list(runner.generate([1,2],max_new_tokens=3))[-1]['finish_reason'],'stop')
            left.stop()
            for thread in threads:thread.join(3)
            self.assertFalse(errors)

    def test_invalid_request_is_rejected_before_any_link_dispatch(self):
        _,owners=self.session();left=owners[0];runner=HostCompletion(left)
        for call in (lambda:left.prefill([1]*5),lambda:left.prefill([59]),lambda:left.decode([1],[0]),
                     lambda:left.prefill_batch([[1],[2]],[0,0]),
                     lambda:list(runner.generate([1]*5,max_new_tokens=1,chunk_size=5))):
            with patch.object(left.channels[0],'begin',side_effect=AssertionError('sent')):
                with self.assertRaises(ValueError):call()
        self.assertTrue(all(w.ready and not w.failed for w in owners))
        self.assertTrue(all(c.state.sequence==1 for c in left.channels))

    def test_admission_precedes_local_layers_and_worker_failure_discards_all_links(self):
        for failed_index in (1,2):
            _,owners=self.session();left=owners[0]
            with patch.object(owners[failed_index].stage,'forward_layers',side_effect=RuntimeError('compute failed')):
                with ThreadPoolExecutor(len(owners)) as pool:
                    futures=[pool.submit(left.prefill,[1,2]),*[pool.submit(w.step) for w in owners[1:]]]
                    for index,future in enumerate(futures):
                        if index==0 or index>=failed_index:
                            with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):future.result(5)
                        else:future.result(5) # Earlier completion is not global success.
            for worker in owners[1:failed_index]:
                with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):worker.step()
            self.assertTrue(all(w.failed and w.closed for w in owners))
            self.assertTrue(left.failed and left.closed and not left.stage.w)
            self.assertTrue(all(c.closed for c in left.channels));self.assertEqual(left.stage.lengths,[0,0])
        _,owners=self.session();left=owners[0];self.pump(owners)
        with patch.object(left.channels[-1],'begin',side_effect=RuntimeError('admission failed')), \
                patch.object(left.stage,'forward_layers',side_effect=AssertionError('computed before admission')):
            with self.assertRaisesRegex(RuntimeError,'admission failed'):left.prefill([1])
        self.assertTrue(all(c.closed for c in left.channels))

    def test_abandoned_completion_and_failed_cleanup_retain_retry_owner(self):
        _,owners=self.session();self.pump(owners);left=owners[0];runner=HostCompletion(left)
        stream=runner.generate([1,2],max_new_tokens=4);next(stream)
        with patch.object(left.stage,'close',side_effect=RuntimeError('cleanup failed')):
            stream.close()
        self.assertTrue(all(c.closed for c in left.channels));self.assertFalse(left.closed)
        runner.cleanup();self.assertTrue(left.closed);self.assertFalse(left.stage.w)

    def test_invalid_constructor_does_not_claim_stage(self):
        source,owners=self.session(ready=False);left=owners[0]
        stage=CpuHybridStage(source,0,1,max_batch=2,**OPTIONS);self.addCleanup(stage.close)
        for channels in (list(reversed(left.channels)),left.channels):
            with self.assertRaises(ValueError):HostPipelineCoordinator(stage,left.topology,channels)
            self.assertIsNone(getattr(stage,'_pipeline_owner',None))
        with self.assertRaises(ValueError):HostPipelineWorker(stage,left.channels[0])
