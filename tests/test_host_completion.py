from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

from dllm.host_completion import HostCompletion
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.hybrid_pipeline import PipelineCancelled
from tests import test_cpu_host_worker_build as fixtures
from tests.test_cpu_pipeline_generation import OPTIONS


class HostCompletionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CpuHostWorkerBuildTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def session(self,mode='none',*,pump=True,eos_ids=()):
        left,right=[self.fixture.build(t,mode) for t in self.fixture.transports()]
        self.fixture.together([left.handshake,right.handshake])
        runner=HostCompletion(left,eos_ids=eos_ids)
        if pump:
            errors=[]
            def serve():
                try:
                    while not right.closed:right.step()
                except (OSError,EOFError,RuntimeError,ValueError) as error:errors.append(error)
                finally:right.close()
            thread=threading.Thread(target=serve,daemon=True);thread.start()
            def close():
                runner.close();right.close();thread.join(3)
                self.assertFalse(thread.is_alive())
            self.addCleanup(close)
        else:self.addCleanup(runner.close)
        return runner,left,right

    def test_ragged_outputs_match_whole_pipeline_and_release_for_reuse(self):
        for mode in ('none','q4_0','q8_0'):
            runner,left,right=self.session(mode)
            model=load_cpu_checkpoint(self.fixture.path,quantization=mode,max_context=16)
            control=CpuPipelineReferenceGenerator(model,[2**22]*2,vocab_size=64,**OPTIONS)
            self.addCleanup(control.cleanup)
            rows=[[1,2,3,4],[5]];limits=[1,4]
            expected=list(control.generate_batch(rows,max_new_tokens=limits,chunk_size=2))
            actual=list(runner.generate_batch(rows,max_new_tokens=limits,chunk_size=2))
            self.assertEqual(actual,expected)
            self.assertEqual(left.channel.state.lengths,[0]*2)
            self.assertEqual(right.channel.state.lengths,[0]*2)
            self.assertEqual(list(runner.generate([4,3],max_new_tokens=3,chunk_size=1)),
                             list(control.generate([4,3],max_new_tokens=3,chunk_size=1)))
            self.assertTrue(left.ready and right.ready)
            runner.close()

    def test_eos_stops_without_decode_and_success_follows_release(self):
        runner,left,right=self.session()
        probe=list(runner.generate([1,2],max_new_tokens=1));token=probe[0]['token_id']
        runner.eos_ids=frozenset([token])
        with patch.object(left,'decode',side_effect=AssertionError('decode after EOS')):
            events=list(runner.generate([1,2],max_new_tokens=4))
        self.assertEqual(events[-1]['finish_reason'],'stop')
        self.assertEqual(events[-1]['usage']['completion_tokens'],1)
        self.assertFalse(any(left.channel.state.lengths));self.assertFalse(any(right.channel.state.lengths))

    def test_invalid_requests_do_not_dispatch_or_discard_session(self):
        runner,left,_=self.session()
        before=left.channel.state.sequence
        for rows,options in (([[1]],{'max_new_tokens':0}),([[True]],{}),([[1]]*5,{}),
                             ([[1]],{'chunk_size':0}),([[1]],{'cancelled':True}),([[63]],{'max_new_tokens':16})):
            with self.assertRaises(ValueError):list(runner.generate_batch(rows,**options))
        self.assertEqual(left.channel.state.sequence,before);self.assertTrue(left.ready)
        self.assertEqual(list(runner.generate([1],max_new_tokens=1))[-1]['finish_reason'],'length')

    def test_constructor_rejects_duplicate_or_bad_stop_policy_without_claim(self):
        transports=self.fixture.transports();left,right=[self.fixture.build(t) for t in transports]
        self.fixture.together([left.handshake,right.handshake])
        for eos in ([True],[64],list(range(257))):
            with self.assertRaises(ValueError):HostCompletion(left,eos_ids=eos)
            self.assertIsNone(getattr(left,'_completion_owner',None))
        HostCompletion(left)
        with self.assertRaises(ValueError):HostCompletion(left)
        with self.assertRaises(ValueError):HostCompletion(right)

    def test_early_consumer_close_discards_epoch(self):
        runner,left,right=self.session()
        stream=runner.generate([1,2],max_new_tokens=4);next(stream);stream.close()
        self.assertTrue(left.closed);self.assertFalse(left.stage.w)
        with self.assertRaises(PipelineCancelled):list(runner.generate([1],max_new_tokens=1))

    def test_cancellation_discards_and_failed_cleanup_is_retryable(self):
        runner,left,_=self.session()
        cancel=threading.Event();stream=runner.generate([1,2],max_new_tokens=4,cancelled=cancel.is_set)
        next(stream);cancel.set()
        with patch.object(left.stage,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaises(PipelineCancelled) as caught:list(stream)
        self.assertTrue(caught.exception.__notes__);self.assertFalse(left.closed);self.assertTrue(left.stage.w)
        runner.cleanup();self.assertTrue(left.closed);self.assertFalse(left.stage.w)

    def test_abort_interrupts_blocked_peer_io(self):
        runner,left,_=self.session(pump=False)
        entered=threading.Event();original=left.prefill_chunk
        def prefill(*args,**kwargs):entered.set();return original(*args,**kwargs)
        with patch.object(left,'prefill_chunk',prefill),ThreadPoolExecutor(1) as pool:
            future=pool.submit(lambda:list(runner.generate([1],max_new_tokens=1)))
            self.assertTrue(entered.wait(2));runner.close()
            with self.assertRaises((OSError,RuntimeError,EOFError)):future.result(timeout=2)
        self.assertTrue(left.closed)

    def test_release_failure_emits_no_final_success(self):
        runner,left,_=self.session();events=[]
        with patch.object(left,'release',side_effect=RuntimeError('release failed')):
            with self.assertRaisesRegex(RuntimeError,'release failed'):
                for event in runner.generate([1],max_new_tokens=1):events.append(event)
        self.assertFalse(any('finish_reason' in e for e in events));self.assertTrue(left.closed)

    def test_concurrent_request_rejected_without_disturbing_owner(self):
        runner,left,_=self.session();stream=runner.generate([1],max_new_tokens=3)
        next(stream)
        with self.assertRaisesRegex(RuntimeError,'active'):list(runner.generate([2],max_new_tokens=1))
        self.assertEqual(list(stream)[-1]['usage']['completion_tokens'],3)
        self.assertTrue(left.ready)
