import unittest
from unittest.mock import patch
import torch
from damendaemon.cpu_pipeline_build import build_cpu_pipeline,CpuPipelineBuildError
from damendaemon.cpu_hybrid_stage import CpuHybridStage
from damendaemon.hybrid_pipeline import HybridPipeline,PipelineCancelled
from tests.test_cpu_hybrid_stage import model,decoded
from tests.test_cpu_stage_plan import OPTIONS


class CpuPipelineBuildTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def build(self,source):
        pipeline=build_cpu_pipeline(source,[2**22,2**22],**OPTIONS);self.addCleanup(pipeline.close);return pipeline

    def test_built_packed_execution_status_and_cancellation(self):
        for mode in ('q4_0','q8_0'):
            source=model(mode);pipeline=self.build(source);control=self.build(decoded(source))
            for tokens,slot,start,final in (([1,2],0,0,False),([3,4],0,2,True),([7,8,9],1,0,True)):
                actual=pipeline.prefill_chunk(tokens,slot,start,final);expected=control.prefill_chunk(tokens,slot,start,final)
                if final:torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
                else:self.assertIsNone(actual)
            actual,ids=pipeline.decode_batch([10,11],[1,0]);expected,expected_ids=control.decode_batch([10,11],[1,0])
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6);torch.testing.assert_close(ids,expected_ids)
            self.assertEqual(pipeline.lengths,control.lengths)
            status=pipeline.status();self.assertTrue(all(s['resident_scale_bytes']>0 for s in status['stages']))
            with self.assertRaises(PipelineCancelled):pipeline.decode_batch([12],[0],cancelled=lambda:True)
            self.assertFalse(pipeline.failed)
            pipeline.close();self.assertTrue(all(s['resident_scale_bytes']==s['resident_weight_bytes']==0 for s in pipeline.status()['stages']))
            self.assertTrue(source.weights);self.assertTrue(source.scales)

    def test_no_fit_precedes_construction(self):
        with patch.object(CpuHybridStage,'__init__',side_effect=AssertionError('constructed')):
            with self.assertRaises(MemoryError):build_cpu_pipeline(model('q4_0'),[1,1],**OPTIONS)

    def test_partial_failure_releases_owned_stages_and_preserves_source(self):
        source=model('q8_0');original=dict(source.weights);initialize=CpuHybridStage.__init__;calls=[]
        def fail(stage,*args,**kwargs):
            calls.append(stage);initialize(stage,*args,**kwargs)
            if len(calls)==2:raise MemoryError('injected second stage')
        with patch.object(CpuHybridStage,'__init__',fail):
            with self.assertRaises(CpuPipelineBuildError) as caught:build_cpu_pipeline(source,[2**22]*2,**OPTIONS)
        error=caught.exception;self.assertTrue(error.closed);self.assertEqual(len(error.stages),2)
        self.assertTrue(all(not stage.w and not stage.model.scales for stage in error.stages));error.close()
        for name,value in original.items():self.assertIs(source.weights[name],value)

    def test_cleanup_failure_retains_retryable_owner(self):
        source=model();initialize=CpuHybridStage.__init__
        def fail(stage,*args,**kwargs):initialize(stage,*args,**kwargs);raise RuntimeError('primary')
        with patch.object(CpuHybridStage,'__init__',fail),patch.object(CpuHybridStage,'release_partial',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(CpuPipelineBuildError,'primary') as caught:build_cpu_pipeline(source,[2**22]*2,**OPTIONS)
        error=caught.exception;self.assertFalse(error.closed);self.assertTrue(error.stages[0].w);self.assertTrue(error.__notes__)
        error.close();self.assertTrue(error.closed);self.assertFalse(error.stages[0].w)

    def test_mid_execution_cancellation_does_not_commit_cursors(self):
        pipeline=self.build(model('q4_0'));calls=[]
        def cancelled():calls.append(1);return len(calls)==3
        with self.assertRaises(PipelineCancelled):pipeline.prefill_chunk([1,2],cancelled=cancelled)
        self.assertTrue(pipeline.failed);self.assertEqual(pipeline.lengths,[0,0])
        self.assertTrue(all(stage.lengths==[0,0] for stage in pipeline.stages))
        with self.assertRaises(RuntimeError):pipeline.prefill([1])

    def test_changed_construction_cost_releases_stage(self):
        initialize=CpuHybridStage.__init__
        def changed(stage,*args,**kwargs):initialize(stage,*args,**kwargs);stage.required_memory_bytes=2**30
        with patch.object(CpuHybridStage,'__init__',changed):
            with self.assertRaisesRegex(CpuPipelineBuildError,'admission changed') as caught:
                build_cpu_pipeline(model(),[2**22]*2,**OPTIONS)
        self.assertTrue(caught.exception.closed);self.assertEqual(len(caught.exception.stages),1)

    def test_failed_or_mixed_quantization_stages_reject_without_claiming(self):
        source=model('q4_0');left=CpuHybridStage(source,0,2,**OPTIONS);right=CpuHybridStage(source,2,4,**OPTIONS)
        self.addCleanup(left.close);self.addCleanup(right.close)
        left._failed=True
        with self.assertRaises(ValueError):HybridPipeline([left,right])
        self.assertIsNone(getattr(right,'_pipeline_owner',None));left._failed=False
        right.model.quantization='q8_0'
        with self.assertRaises(ValueError):HybridPipeline([left,right])
        self.assertIsNone(getattr(left,'_pipeline_owner',None))
