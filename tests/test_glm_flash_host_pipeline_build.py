from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.cpu_snapshot_identity import fingerprint
from damendaemon.glm_flash_host_pipeline_build import build_glm_flash_host_pipeline_worker,FlashHostPipelineBuildError
from damendaemon.host_topology import HostTopology
from tests import test_cpu_host_pipeline_build as hybrid_tests
from tests.test_glm_flash_reference_loader import fixture,floating_control


class FlashHostPipelineBuildTests(unittest.TestCase):
    links=hybrid_tests.CpuHostPipelineBuildTests.links
    together=hybrid_tests.CpuHostPipelineBuildTests.together

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,_,self.values=fixture(self.path)
        (self.path/'tokenizer.json').write_text('{}')

    def plan(self,verified=True,config=None):
        identity=bytes.fromhex(fingerprint(self.path).model_id) if verified else b's'*32
        return HostTopology(config or self.config,[(0,1),(1,2),(2,3)],identity,quantization='fp8_reference',
                            context=16,slots=2,chunk_size=4,frame_bytes=4096)

    def build(self,plan,index,links,**options):
        worker=build_glm_flash_host_pipeline_worker(self.path,plan,index,links,epoch=1,**options)
        self.addCleanup(worker.close);return worker

    def test_verified_stage_local_workers_match_whole_control_and_release_owned_weights(self):
        plan=self.plan();links=self.links(plan)
        workers=[self.build(plan,index,group,verify_source=True) for index,group in enumerate(links)]
        for index,worker in enumerate(workers):
            self.assertTrue(worker.model_identity_verified)
            self.assertEqual(worker.source_model_id,plan.source_identity.hex())
            model=worker.stage.stage
            self.assertEqual((model.start,model.end),plan.ranges[index])
            self.assertEqual(model.embedding is not None,index==0)
            self.assertEqual(model.head is not None,index==2)
        self.together([worker.handshake for worker in workers])
        control=floating_control(self.config,self.values)
        logits,state=control.prefill(torch.tensor([[1,2,3]]),capacity=16);state.close()
        actual=self.together([lambda:workers[0].prefill([1,2,3]),*[worker.step for worker in workers[1:]]])[0]
        self.assertEqual(actual,int(logits.argmax(-1)))
        self.together([lambda:workers[0].release([0]),*[worker.step for worker in workers[1:]]])
        self.together([workers[0].stop,*[worker.step for worker in workers[1:]]])
        self.assertTrue(all(worker.closed and worker.stage.stage is None for worker in workers))
        self.assertEqual(workers[0].stage.c,self.config)

    def test_invalid_controls_leave_links_unclaimed(self):
        plan=self.plan();links=self.links(plan)[0]
        for options in (dict(verify_source=1),dict(max_cache_bytes=0),dict(identity_timeout=float('nan')),dict(epoch=0)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                build_glm_flash_host_pipeline_worker(self.path,plan,0,links,**(dict(epoch=1)|options))
            self.assertTrue(all(not link.closed and getattr(link,'_cpu_worker_builder_owner',None) is None for link in links))
        with self.assertRaises(ValueError):self.build(plan,0,links[::-1])
        self.assertTrue(all(not link.closed for link in links))

    def test_hash_config_and_weight_budget_failure_close_every_claimed_link(self):
        for kind in ('hash','config','budget'):
            with self.subTest(kind=kind):
                plan=self.plan(verified=False,config=self.config|dict(text_config=self.config['text_config']|dict(rms_norm_eps=2e-5)) if kind=='config' else None)
                links=self.links(plan)[0]
                options=dict(verify_source=True) if kind=='hash' else dict(max_weight_bytes=1) if kind=='budget' else {}
                with self.assertRaises(FlashHostPipelineBuildError) as failure:self.build(plan,0,links,**options)
                self.assertTrue(failure.exception.closed)
                self.assertTrue(all(link.closed for link in links))

    def test_snapshot_change_after_loading_or_slot_construction_rejected(self):
        import damendaemon.glm_flash_host_pipeline_build as builder
        for target in ('load_reference_stage','_OwnedFlashHostStage'):
            with self.subTest(target=target):
                plan=self.plan();links=self.links(plan)[0];original=getattr(builder,target)
                def mutate(*args,**kwargs):
                    result=original(*args,**kwargs)
                    file=self.path/'part-1.safetensors';replacement=self.path/'replacement'
                    replacement.write_bytes(file.read_bytes());replacement.replace(file)
                    return result
                with patch.object(builder,target,mutate):
                    with self.assertRaises(FlashHostPipelineBuildError) as failure:self.build(plan,0,links,verify_source=True)
                self.assertTrue(failure.exception.closed);self.assertTrue(all(link.closed for link in links))
                if failure.exception.stage is not None:self.assertIsNone(failure.exception.stage.stage)

    def test_partial_transport_close_failure_retains_retry_owner(self):
        plan=self.plan(False);links=self.links(plan)[0]
        with patch.object(links[0],'close',side_effect=OSError('close')):
            with self.assertRaises(FlashHostPipelineBuildError) as failure:self.build(plan,0,links,max_weight_bytes=1)
        self.assertFalse(failure.exception.closed)
        self.assertTrue(links[1].closed);self.assertTrue(failure.exception.__notes__)
        failure.exception.close();self.assertTrue(failure.exception.closed);self.assertTrue(links[0].closed)
