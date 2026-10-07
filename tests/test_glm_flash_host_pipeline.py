from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_stage_slots import FlashStageSlots
from dllm.glm_flash_reference_loader import load_reference_stage
from dllm.host_topology import HostTopology
from dllm.host_pipeline_worker import HostPipelineCoordinator,HostPipelineWorker
from dllm.host_tensor_channel import HostTensorChannel
from dllm.planned_host_stage_channel import PlannedHostStageChannel
from dllm.stage_protocol import StageState
from tests import test_host_tls,test_glm_host_pipeline as glm_tests
from tests.test_glm_flash_reference_loader import fixture,floating_control


class FlashHostPipelineTests(unittest.TestCase):
    command=glm_tests.GlmHostPipelineTests.command
    pump=glm_tests.GlmHostPipelineTests.pump
    expected=glm_tests.GlmHostPipelineTests.expected
    test_real_two_three_stage_tls_ragged_chunks_decode_release_and_stop=glm_tests.GlmHostPipelineTests.test_real_two_three_stage_tls_ragged_chunks_decode_release_and_stop
    test_middle_failure_discards_all_links_and_owned_slot_caches=glm_tests.GlmHostPipelineTests.test_middle_failure_discards_all_links_and_owned_slot_caches
    test_peer_loss_and_abandoned_completion_discard_epoch=glm_tests.GlmHostPipelineTests.test_peer_loss_and_abandoned_completion_discard_epoch

    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,_,self.values=fixture(self.path)

    def plan(self,config=None,count=3,**options):
        return HostTopology(config or self.config,[(0,1),(1,2),(2,3)] if count==3 else [(0,1),(1,3)],b'f'*32,
            **(dict(quantization='fp8_reference',context=16,slots=2,chunk_size=8,frame_bytes=4096,mapped_vocab_size=30,excluded_ids=[29])|options))

    def session(self,count=2):
        topology=self.plan(count=count);stages=[FlashStageSlots(load_reference_stage(self.path,a,b),16,2) for a,b in topology.ranges]
        for stage in stages:self.addCleanup(stage.close)
        links=[];workers=[]
        for index in range(1,count):
            fixture_tls=test_host_tls.HostTlsTests();self.addCleanup(fixture_tls.doCleanups);channels=[]
            for rank,sock in enumerate(fixture_tls.pair()):
                transport=HostTensorChannel(sock,topology.stage_identity(index),rank,max_frame_bytes=4096,timeout=2)
                self.addCleanup(transport.close);channels.append(PlannedHostStageChannel(transport,StageState(11,16,2),topology,index))
            links.append(channels[0]);workers.append(HostPipelineWorker(stages[index],channels[1]))
        owners=[HostPipelineCoordinator(stages[0],topology,links),*workers]
        for owner in owners:self.addCleanup(owner.close)
        with ThreadPoolExecutor(count) as pool:
            futures=[pool.submit(owner.handshake) for owner in owners]
            for future in futures:future.result(5)
        return topology,owners,floating_control(self.config,self.values)

    def test_full_stream_frame_budget_identity_and_explicit_admission(self):
        plan=self.plan();self.assertEqual(plan.width,32);self.assertEqual(plan.text_config,self.config['text_config'])
        for options in (dict(quantization='none'),dict(frame_bytes=512),dict(frame_bytes=1023),dict(chunk_size=True)):
            with self.assertRaises(ValueError):self.plan(**options)
        self.assertEqual(self.plan(frame_bytes=1024).width,32)
        config=deepcopy(self.config);config['text_config']['hc_mult']=3
        self.assertNotEqual(plan.identity,self.plan(config).identity)
        config=deepcopy(self.config);config['image_token_id']=28
        changed=self.plan(config);self.assertIn(28,changed.excluded_ids);self.assertNotEqual(plan.identity,changed.identity)
        config=deepcopy(self.config);config['text_config']['multimodal_token_ids']=None
        with self.assertRaises(ValueError):self.plan(config)
        from dllm.glm_host_pipeline_build import build_glm_host_pipeline_worker
        with self.assertRaises(ValueError):build_glm_host_pipeline_worker('/unused',plan,0,[],epoch=1)
        _,owners,_=self.session()
        from dllm.host_serving import HostServingTopology
        with self.assertRaises(ValueError):HostServingTopology._validate_topology(None,owners[0],4)
        with self.assertRaises(ValueError):owners[0].prefill([29])
        self.assertTrue(owners[0].ready);self.command(owners,owners[0].stop)

    def test_mismatched_stream_contract_rejects_tls_handshake(self):
        config=deepcopy(self.config);config['text_config']['hc_mult']=3
        plans=[self.plan(),self.plan(config)]
        fixture_tls=test_host_tls.HostTlsTests();self.addCleanup(fixture_tls.doCleanups);transports=[]
        for rank,sock in enumerate(fixture_tls.pair()):
            transport=HostTensorChannel(sock,plans[rank].stage_identity(1),rank,max_frame_bytes=4096,timeout=2)
            self.addCleanup(transport.close);transports.append(transport)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(t.handshake) for t in transports]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,EOFError,OSError)):future.result(5)
