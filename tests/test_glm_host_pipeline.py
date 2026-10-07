from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_stage_slots import GlmStageSlots
from dllm.glm_reference_loader import load_reference_stage
from dllm.host_topology import HostTopology
from dllm.host_pipeline_worker import HostPipelineCoordinator,HostPipelineWorker
from dllm.host_tensor_channel import HostTensorChannel
from dllm.planned_host_stage_channel import PlannedHostStageChannel
from dllm.stage_protocol import StageState
from tests import test_host_tls,test_host_pipeline_worker
from tests.test_glm_checkpoint import config as geometry
from tests.test_glm_reference_loader import fixture,floating_control


class GlmHostPipelineTests(unittest.TestCase):
    command=test_host_pipeline_worker.HostPipelineWorkerTests.command
    pump=test_host_pipeline_worker.HostPipelineWorkerTests.pump

    def session(self,count=2):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)
        config=geometry()|dict(rms_norm_eps=1e-5,routed_scaling_factor=2.5)
        if count==3:config|=dict(num_hidden_layers=4,indexer_types=['full','shared','full','full'],mlp_layer_types=['dense','sparse','sparse','sparse'])
        with patch('tests.test_glm_reference_loader.geometry',return_value=config):config,_,values=fixture(path)
        ranges=[(0,2),(2,3)] if count==2 else [(0,2),(2,3),(3,4)]
        topology=HostTopology(config,ranges,b'g'*32,quantization='fp8_reference',context=16,
            slots=2,chunk_size=8,frame_bytes=4096,mapped_vocab_size=30,excluded_ids=[29])
        stages=[GlmStageSlots(load_reference_stage(path,a,b),16,2) for a,b in ranges]
        links=[];workers=[]
        for stage in stages:self.addCleanup(stage.close)
        for index in range(1,count):
            fixture_tls=test_host_tls.HostTlsTests();self.addCleanup(fixture_tls.doCleanups)
            channels=[]
            for rank,sock in enumerate(fixture_tls.pair()):
                transport=HostTensorChannel(sock,topology.stage_identity(index),rank,max_frame_bytes=4096,timeout=2)
                self.addCleanup(transport.close)
                channels.append(PlannedHostStageChannel(transport,StageState(11,16,2),topology,index))
            links.append(channels[0]);workers.append(HostPipelineWorker(stages[index],channels[1]))
        owners=[HostPipelineCoordinator(stages[0],topology,links),*workers]
        for owner in owners:self.addCleanup(owner.close)
        with ThreadPoolExecutor(count) as pool:
            futures=[pool.submit(owner.handshake) for owner in owners]
            for future in futures:future.result(5)
        return topology,owners,floating_control(config,values)

    def expected(self,control,tokens):
        logits,state=control.prefill(torch.tensor([tokens]),capacity=16)
        state.close();scores=logits[:,:30].clone();scores[:,29]=-float('inf')
        return int(scores.argmax(-1))

    def test_real_two_three_stage_tls_ragged_chunks_decode_release_and_stop(self):
        for count in (2,3):
            with self.subTest(count=count):
                topology,owners,control=self.session(count);first=owners[0]
                prompts=[[1,2],[3,4,2]]
                original=owners[-1].stage.output_logits
                def compare(hidden):
                    actual=original(hidden);expected=[]
                    for row in prompts:
                        logits,state=control.prefill(torch.tensor([row]),capacity=16)
                        state.close();expected.append(logits)
                    torch.testing.assert_close(actual,torch.cat(expected),rtol=3e-5,atol=3e-6)
                    return actual
                with patch.object(owners[-1].stage,'output_logits',compare):
                    actual=self.command(owners,lambda:first.prefill_batch(prompts,[1,0]))
                self.assertEqual(actual,[self.expected(control,row) for row in prompts])
                actual=self.command(owners,lambda:first.decode([4,5],[0,1]))
                self.assertEqual(actual,[self.expected(control,prompts[1]+[4]),self.expected(control,prompts[0]+[5])])
                self.command(owners,lambda:first.release([0,1]))
                self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))
                self.command(owners,lambda:first.prefill_chunk([1,2],1,0))
                actual=self.command(owners,lambda:first.prefill_chunk([3,4],1,2))
                self.assertEqual(actual,self.expected(control,[1,2,3,4]))
                self.command(owners,lambda:first.release([1]))
                self.command(owners,first.stop)
                self.assertTrue(all(owner.closed for owner in owners))

    def test_topology_precision_cuts_and_existing_builder_http_admission(self):
        config=geometry()|dict(rms_norm_eps=1e-5,routed_scaling_factor=2.5);ranges=[(0,2),(2,3)]
        for options in (dict(quantization='none'),dict(quantization='q8_0'),dict(quantization='fp8_reference',ranges=[(0,1),(1,3)])):
            options=dict(ranges=ranges)|options
            with self.assertRaises(ValueError):HostTopology(config,source_identity=b'g'*32,context=16,**options)
        topology,owners,_=self.session()
        from dllm.cpu_host_pipeline_build import build_cpu_host_pipeline_worker
        with self.assertRaisesRegex(ValueError,'controls'):
            build_cpu_host_pipeline_worker('/unused',topology,0,[],epoch=1)
        from dllm.host_serving import HostServingTopology
        with self.assertRaises(ValueError):HostServingTopology._validate_topology(None,owners[0],4)
        with self.assertRaises(ValueError):owners[0].prefill([29])
        self.assertTrue(owners[0].ready)
        self.command(owners,owners[0].stop)

    def test_middle_failure_discards_all_links_and_owned_slot_caches(self):
        _,owners,_=self.session(3)
        with patch.object(owners[1].stage,'forward_layers',side_effect=RuntimeError('middle')):
            self.command(owners,lambda:owners[0].prefill([1,2]),fails=True)
        self.assertTrue(all(owner.failed for owner in owners))
        self.assertTrue(all(owner.stage.closed for owner in owners))
        self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))

    def test_peer_loss_and_abandoned_completion_discard_epoch(self):
        from dllm.host_completion import HostCompletion
        _,owners,_=self.session();errors,threads=self.pump(owners)
        runner=HostCompletion(owners[0],eos_ids=[])
        stream=runner.generate([1,2],max_new_tokens=5,chunk_size=2);next(stream);stream.close()
        self.assertTrue(runner.stopped.is_set());self.assertTrue(owners[0].closed)
        for thread in threads:thread.join(3);self.assertFalse(thread.is_alive())
        self.assertTrue(all(owner.stage.closed for owner in owners))
        _,owners,_=self.session();errors,threads=self.pump(owners)
        runner=HostCompletion(owners[0],eos_ids=[])
        self.assertTrue(list(runner.generate([2,3],max_new_tokens=1,chunk_size=2)))
        owners[1].close()
        with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):list(runner.generate([1],max_new_tokens=1,chunk_size=2))
        self.assertTrue(owners[0].failed)

    def test_glm_token_policy_identity_disagreement_fails_tls_setup(self):
        config=geometry()|dict(rms_norm_eps=1e-5,routed_scaling_factor=2.5);ranges=[(0,2),(2,3)]
        plans=[HostTopology(config,ranges,b'g'*32,quantization='fp8_reference',context=16,
                           slots=2,chunk_size=8,frame_bytes=4096,mapped_vocab_size=size) for size in (30,31)]
        self.assertNotEqual(plans[0].identity,plans[1].identity)
        fixture_tls=test_host_tls.HostTlsTests();self.addCleanup(fixture_tls.doCleanups)
        transports=[]
        for rank,sock in enumerate(fixture_tls.pair()):
            transport=HostTensorChannel(sock,plans[rank].stage_identity(1),rank,max_frame_bytes=4096,timeout=2)
            self.addCleanup(transport.close);transports.append(transport)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(transport.handshake) for transport in transports]
            for future in futures:
                with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):future.result(5)
