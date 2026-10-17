from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.config import hybrid_config
from damendaemon.cpu_host_pipeline_build import build_cpu_host_pipeline_worker,CpuHostPipelineBuildError
from damendaemon.cpu_hybrid_checkpoint import load_cpu_checkpoint,load_cpu_stage_checkpoint
from damendaemon.cpu_hybrid_stage import CpuHybridStage
from damendaemon.cpu_snapshot_identity import fingerprint
from damendaemon.host_topology import HostTopology
from damendaemon.host_tensor_channel import HostTensorChannel
from tests.test_cpu_hybrid_checkpoint import snapshot
from tests.test_cpu_pipeline_generation import OPTIONS
from tests import test_host_tls


class CpuHostPipelineBuildTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        snapshot(self.path);(self.path/'tokenizer.json').write_text('{}')
        self.config=hybrid_config(json.loads((self.path/'config.json').read_text()))

    def plan(self,mode='none',verified=False,**options):
        identity=bytes.fromhex(fingerprint(self.path).model_id) if verified else b't'*32
        return HostTopology(self.config,[(0,1),(1,3),(3,4)],identity,quantization=mode,
                            context=16,slots=2,chunk_size=4,frame_bytes=4096,**options)

    def links(self,plan):
        links=[[] for _ in plan.ranges]
        for index in range(1,len(plan.ranges)):
            fixture=test_host_tls.HostTlsTests();self.addCleanup(fixture.doCleanups)
            for rank,sock in enumerate(fixture.pair()):
                transport=HostTensorChannel(sock,plan.stage_identity(index),rank,max_frame_bytes=4096,timeout=2)
                self.addCleanup(transport.close);links[0 if rank==0 else index].append(transport)
        return links

    def build(self,plan,index,links,**options):
        worker=build_cpu_host_pipeline_worker(self.path,plan,index,links,epoch=1,**(OPTIONS|options))
        self.addCleanup(worker.close);return worker

    def together(self,calls):
        with ThreadPoolExecutor(len(calls)) as pool:
            return [f.result(5) for f in [pool.submit(call) for call in calls]]

    def test_stage_local_snapshot_workers_match_whole_checkpoint_over_tls(self):
        for mode in ('none','q4_0','q8_0'):
            plan=self.plan(mode,True);links=self.links(plan)
            workers=[self.build(plan,index,group,verify_source=True) for index,group in enumerate(links)]
            for index,worker in enumerate(workers):
                a,b=plan.ranges[index]
                self.assertTrue(worker.model_identity_verified)
                self.assertEqual(worker.source_model_id,plan.source_identity.hex())
                for name in worker.stage.w:
                    if '.' in name:self.assertTrue(a<=int(name.split('.')[0])<b)
                    elif name=='embed':self.assertEqual(index,0)
                    else:self.assertEqual(index,2)
            self.together([w.handshake for w in workers]);left=workers[0]
            source=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
            full=CpuHybridStage(source,0,4,max_batch=2,**OPTIONS);self.addCleanup(full.close)
            ids=torch.tensor([[1,2,3]])
            expected=full.output_logits(full.forward_layers(full.embedding(ids),torch.tensor([[0,1,2]]),torch.tensor([0]),True)[:,-1]).argmax(-1).item()
            actual=self.together([lambda:left.prefill([1,2,3]),*[w.step for w in workers[1:]]])[0]
            self.assertEqual(actual,expected)
            self.together([left.stop,*[w.step for w in workers[1:]]])
            self.assertTrue(all(w.closed for w in workers))

    def test_invalid_control_does_not_claim_any_transport(self):
        plan=self.plan();links=self.links(plan)[0]
        cases=[dict(index=0,transports=links,epoch=0),dict(index=0,transports=list(reversed(links)),epoch=1),
               dict(index=1,transports=links[:1],epoch=1),dict(index=0,transports=links,epoch=1,verify_source=1)]
        for options in cases:
            with patch('damendaemon.cpu_host_pipeline_build.load_cpu_stage_checkpoint',side_effect=AssertionError('loaded')):
                with self.assertRaises(ValueError):build_cpu_host_pipeline_worker(self.path,plan,**options)
            self.assertTrue(all(not t.closed and getattr(t,'_cpu_worker_builder_owner',None) is None for t in links))
        worker=self.build(plan,0,links)
        self.assertFalse(worker.model_identity_verified)
        with self.assertRaises(ValueError):self.build(plan,0,links)
        self.assertFalse(worker.closed)

    def test_mismatched_identity_or_configuration_closes_claimed_links(self):
        for verified in (False,True):
            plan=self.plan();links=self.links(plan)[0]
            if not verified:
                config=json.loads((self.path/'config.json').read_text());config['rms_norm_eps']=2e-6
                (self.path/'config.json').write_text(json.dumps(config))
            with self.assertRaises(CpuHostPipelineBuildError) as caught:self.build(plan,0,links,verify_source=verified)
            self.assertTrue(caught.exception.closed);self.assertTrue(all(t.closed for t in links))

    def test_verified_snapshot_changed_during_load_fails_before_stage_allocation(self):
        plan=self.plan(verified=True);links=self.links(plan)[1]
        def changed(*args,**kwargs):
            result=load_cpu_stage_checkpoint(*args,**kwargs)
            (self.path/'tokenizer.json').write_text('{"changed":true}')
            return result
        with patch('damendaemon.cpu_host_pipeline_build.load_cpu_stage_checkpoint',side_effect=changed), \
                patch.object(CpuHybridStage,'__init__',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(CpuHostPipelineBuildError,'changed') as caught:self.build(plan,1,links,verify_source=True)
        self.assertTrue(caught.exception.closed);self.assertTrue(links[0].closed)

    def test_partial_stage_failure_cleans_all_links_and_retains_retryable_owner(self):
        plan=self.plan();links=self.links(plan)[0];initialize=CpuHybridStage.__init__
        def failed(stage,*args,**kwargs):initialize(stage,*args,**kwargs);raise RuntimeError('primary allocation failed')
        with patch.object(CpuHybridStage,'__init__',failed), \
                patch.object(CpuHybridStage,'release_partial',side_effect=RuntimeError('secondary cleanup failed')), \
                patch.object(links[0],'close',side_effect=RuntimeError('first link cleanup failed')):
            with self.assertRaisesRegex(CpuHostPipelineBuildError,'primary allocation failed') as caught:self.build(plan,0,links)
            self.assertTrue(links[1].closed);self.assertFalse(caught.exception.closed)
            self.assertTrue(caught.exception.stage.w);self.assertTrue(caught.exception.__notes__)
        caught.exception.close();self.assertTrue(caught.exception.closed)
        self.assertTrue(all(t.closed for t in links));self.assertFalse(caught.exception.stage.w)

    def test_budget_failure_and_changed_snapshot_after_stage_creation_release_storage(self):
        plan=self.plan();links=self.links(plan)[2]
        with self.assertRaises(CpuHostPipelineBuildError) as caught:self.build(plan,2,links,max_weight_bytes=1)
        self.assertTrue(caught.exception.closed)
        plan=self.plan(verified=True);links=self.links(plan)[2];initialize=CpuHybridStage.__init__;stages=[]
        def changed(stage,*args,**kwargs):
            initialize(stage,*args,**kwargs);stages.append(stage)
            (self.path/'tokenizer.json').write_text('{"changed":true}')
        with patch.object(CpuHybridStage,'__init__',changed):
            with self.assertRaisesRegex(CpuHostPipelineBuildError,'changed') as caught:self.build(plan,2,links,verify_source=True)
        self.assertTrue(caught.exception.closed);self.assertFalse(stages[0].w)
