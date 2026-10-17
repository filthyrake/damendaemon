from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.deepseek_stage_slots import DeepSeekStageSlots
from damendaemon.deepseek_reference_loader import load_reference_stage,load_reference_model
from damendaemon.host_topology import HostTopology
from damendaemon.host_pipeline_worker import HostPipelineCoordinator,HostPipelineWorker
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.planned_host_stage_channel import PlannedHostStageChannel
from damendaemon.stage_protocol import StageState
from tests import test_host_tls,test_glm_host_pipeline as glm_tests
from tests.test_deepseek_reference_loader import fixture
from tests.test_deepseek_checkpoint import config as geometry


class DeepSeekHostPipelineTests(unittest.TestCase):
    command=glm_tests.GlmHostPipelineTests.command
    pump=glm_tests.GlmHostPipelineTests.pump
    test_middle_failure_discards_all_links_and_owned_slot_caches=glm_tests.GlmHostPipelineTests.test_middle_failure_discards_all_links_and_owned_slot_caches
    test_peer_loss_and_abandoned_completion_discard_epoch=glm_tests.GlmHostPipelineTests.test_peer_loss_and_abandoned_completion_discard_epoch

    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        with patch('tests.test_deepseek_reference_loader.geometry',return_value=geometry()|dict(num_hash_layers=2)):self.config,_=fixture(self.path)

    def plan(self,config=None,count=3,**options):
        return HostTopology(config or self.config,[(0,1),(1,2),(2,3)] if count==3 else [(0,1),(1,3)],b'd'*32,
            **(dict(quantization='fp4_fp8_reference',context=140,slots=2,chunk_size=128,frame_bytes=32768,mapped_vocab_size=15,excluded_ids=[14])|options))

    def session(self,count=2):
        topology=self.plan(count=count);stages=[DeepSeekStageSlots(load_reference_stage(self.path,a,b),140,2) for a,b in topology.ranges]
        for stage in stages:self.addCleanup(stage.close)
        links=[];workers=[]
        for index in range(1,count):
            tls=test_host_tls.HostTlsTests();self.addCleanup(tls.doCleanups);channels=[]
            for rank,sock in enumerate(tls.pair()):
                transport=HostTensorChannel(sock,topology.stage_identity(index),rank,max_frame_bytes=32768,timeout=2)
                self.addCleanup(transport.close);channels.append(PlannedHostStageChannel(transport,StageState(11,140,2),topology,index))
            links.append(channels[0]);workers.append(HostPipelineWorker(stages[index],channels[1]))
        owners=[HostPipelineCoordinator(stages[0],topology,links),*workers]
        for owner in owners:self.addCleanup(owner.close)
        with ThreadPoolExecutor(count) as pool:
            futures=[pool.submit(owner.handshake) for owner in owners]
            for future in futures:future.result(5)
        return topology,owners,load_reference_model(self.path)

    def expected(self,control,tokens):
        logits,state=control.prefill(torch.tensor([tokens]),max_tokens=140);state.close()
        scores=logits[:,:15].clone();scores[:,14]=-float('inf');return logits,int(scores.argmax(-1))

    def test_real_two_three_stage_tls_packed_hash_routing_chunks_unequal_decode_and_release(self):
        for count in (2,3):
            _,owners,control=self.session(count);first=owners[0];prompts=[[1,2],[3,4,2]]
            actual=self.command(owners,lambda:first.prefill_batch(prompts,[1,0]))
            self.assertEqual(actual,[self.expected(control,p)[1] for p in prompts])
            original=owners[-1].stage.output_logits
            def compare(hidden):
                actual=original(hidden);expected=torch.cat([self.expected(control,prompts[1]+[4])[0],self.expected(control,prompts[0]+[5])[0]])
                torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6);return actual
            with patch.object(owners[-1].stage,'output_logits',compare):actual=self.command(owners,lambda:first.decode([4,5],[0,1]))
            self.assertEqual(actual,[self.expected(control,prompts[1]+[4])[1],self.expected(control,prompts[0]+[5])[1]])
            self.command(owners,lambda:first.release([0,1]));self.assertTrue(all(o.stage.states==[None,None] for o in owners))
            tokens=[i%14 for i in range(130)]
            self.command(owners,lambda:first.prefill_chunk(tokens[:3],1,0))
            actual=self.command(owners,lambda:first.prefill_chunk(tokens[3:129],1,3))
            self.assertEqual(actual,self.expected(control,tokens[:129])[1])
            self.assertEqual(self.command(owners,lambda:first.decode(tokens[129:],[1])),[self.expected(control,tokens)[1]])
            self.command(owners,lambda:first.release([1]));self.command(owners,first.stop)
            self.assertTrue(all(o.closed for o in owners))

    def test_explicit_quantization_token_identity_and_full_stream_frame_budget(self):
        plan=self.plan();self.assertTrue(plan.routing_tokens);self.assertEqual(plan.width,64)
        for options in (dict(quantization='none'),dict(quantization='fp8_reference'),dict(frame_bytes=32767)):
            with self.assertRaises(ValueError):self.plan(**options)
        with self.assertRaises(ValueError):self.plan(self.config|dict(scoring_func='softmax'))
        config=deepcopy(self.config);config['hc_mult']=3
        changed=self.plan(config,frame_bytes=49152);self.assertNotEqual(changed.identity,plan.identity)
        self.assertNotEqual(plan.identity,self.plan(excluded_ids=[13]).identity)
        from damendaemon.glm_flash_host_pipeline_build import build_glm_flash_host_pipeline_worker
        with self.assertRaises(ValueError):build_glm_flash_host_pipeline_worker('/unused',plan,0,[],epoch=1)
        _,owners,_=self.session()
        with self.assertRaises(ValueError):owners[0].prefill([14])
        self.assertTrue(owners[0].ready);self.command(owners,owners[0].stop)

    def test_bad_routing_frame_shape_dtype_and_policy_fail_before_remote_compute(self):
        for kind in ('shape','dtype','negative','excluded','unmapped'):
            _,owners,_=self.session();transport=owners[0].channels[0].transport;send=transport.send
            def corrupt(tensor):
                if tensor.dtype==torch.int64 and tensor.ndim==1 and tensor.numel()==2:
                    if kind=='shape':tensor=tensor[:1]
                    elif kind=='dtype':tensor=tensor.to(torch.int32)
                    else:tensor=torch.full_like(tensor,{'negative':-1,'excluded':14,'unmapped':15}[kind])
                return send(tensor)
            with patch.object(transport,'send',corrupt),patch.object(owners[1].stage,'forward_layers',side_effect=AssertionError('remote compute')) as compute:
                self.command(owners,lambda:owners[0].prefill([1,2]),fails=True);compute.assert_not_called()
            self.assertTrue(all(o.failed and o.stage.closed for o in owners))

    def test_mismatched_token_policy_rejects_handshake(self):
        plans=[self.plan(),self.plan(excluded_ids=[13])];tls=test_host_tls.HostTlsTests();self.addCleanup(tls.doCleanups);links=[]
        for rank,sock in enumerate(tls.pair()):
            link=HostTensorChannel(sock,plans[rank].stage_identity(1),rank,max_frame_bytes=32768,timeout=2);self.addCleanup(link.close);links.append(link)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(link.handshake) for link in links]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,EOFError,OSError)):future.result(5)
