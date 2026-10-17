from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_layer_pipeline import FlashLayerPipeline
from damendaemon.glm_layer_pipeline import GlmLayerPipeline
from damendaemon.glm_flash_reference_loader import load_reference_stage
from damendaemon.glm_flash_model import FlashModelState
from tests.test_glm_flash_reference_loader import fixture,floating_control


class FlashLayerPipelineTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,_,self.values=fixture(self.path)

    def pipeline(self,ranges=((0,1),(1,3))):
        pipeline=FlashLayerPipeline([load_reference_stage(self.path,a,b) for a,b in ranges])
        self.addCleanup(pipeline.cleanup);return pipeline

    def state(self,pipeline,**options):
        state=pipeline.new_state(2,6,**options);self.addCleanup(state.close);return state

    def test_chunked_reordered_two_three_stage_control_and_distinct_admission(self):
        tokens=torch.tensor([[1,2,3],[3,2,1]]);control=floating_control(self.config,self.values)
        expected,other=control.prefill(tokens,capacity=6,all_logits=True);other.close()
        for ranges in (((0,1),(1,3)),((0,1),(1,2),(2,3))):
            pipeline=self.pipeline(ranges);state=self.state(pipeline)
            self.assertNotIsInstance(pipeline,GlmLayerPipeline)
            actual=pipeline.forward(tokens[:,:2],state,position=0,all_logits=True)
            torch.testing.assert_close(actual,expected[:,:2],rtol=3e-5,atol=3e-6)
            order=torch.tensor([1,0,1]);state.reorder(order)
            actual=pipeline.forward(tokens[order,2:],state,position=2)
            torch.testing.assert_close(actual,expected[order,2],rtol=3e-5,atol=3e-6)
            self.assertEqual(state.batch,3);self.assertEqual(state.position,3)
            with self.assertRaises(ValueError):GlmLayerPipeline(pipeline.stages)
        stages=list(self.pipeline().stages)
        for bad in ([stages[0]],stages[::-1],[stages[0],stages[0],stages[1]]):
            with self.assertRaises(ValueError):FlashLayerPipeline(bad)

    def test_global_mixed_cache_and_residual_boundary_preflight(self):
        pipeline=self.pipeline();per=pipeline.cache_bytes_per_row(6)
        with patch.object(pipeline.stages[0],'new_state',side_effect=AssertionError('allocation')):
            with self.assertRaisesRegex(ValueError,'aggregate'):pipeline.new_state(2,6,max_cache_bytes=2*per-1)
        state=self.state(pipeline,max_cache_bytes=2*per)
        with self.assertRaisesRegex(ValueError,'aggregate'):state.reorder(torch.tensor([1,0,1]))
        self.assertFalse(state.poisoned);self.assertEqual(state.batch,2)
        with self.assertRaises(ValueError):state.reorder(torch.zeros(129,dtype=torch.int64))
        self.assertFalse(state.poisoned)
        # Enough space for [batch, token, width], but not the full stream axis.
        pipeline.stages[-1].max_output_bytes=2*16*4
        with patch.object(pipeline.stages[0],'forward',side_effect=AssertionError('execute')):
            with self.assertRaisesRegex(ValueError,'output budget'):
                pipeline.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)

    def test_partial_forward_reorder_and_close_failures_retain_all_owners(self):
        pipeline=self.pipeline();state=self.state(pipeline)
        with patch.object(pipeline.stages[-1],'forward',side_effect=RuntimeError('later stage')):
            with self.assertRaisesRegex(RuntimeError,'later stage'):pipeline.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertTrue(state.poisoned);self.assertTrue(all(s.poisoned for s in state.stages));state.close()
        state=self.state(pipeline)
        with patch.object(state.stages[-1],'reorder',side_effect=RuntimeError('later reorder')):
            with self.assertRaisesRegex(RuntimeError,'later reorder'):state.reorder(torch.tensor([1,0,1]))
        self.assertTrue(state.poisoned);first,last=state.stages
        with patch.object(first,'close',side_effect=RuntimeError('first close')):
            with self.assertRaisesRegex(RuntimeError,'first close'):state.close()
            self.assertEqual(state.stages,[first]);self.assertTrue(last.closed);self.assertIsNotNone(state.owner)
        state.close();self.assertIsNone(state.owner)

    def test_allocation_and_prefill_cleanup_failure_block_new_cache_until_retry(self):
        pipeline=self.pipeline();last=pipeline.stages[-1]
        with patch.object(last,'new_state',side_effect=MemoryError('allocation')),patch.object(FlashModelState,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaises(MemoryError):pipeline.new_state(2,6)
            self.assertTrue(pipeline.cleanup_pending)
            with self.assertRaisesRegex(RuntimeError,'cleanup'):pipeline.new_state(2,6)
        pipeline.cleanup();self.assertFalse(pipeline.cleanup_pending)
        with patch.object(last,'forward',side_effect=RuntimeError('execute')),patch.object(FlashModelState,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'execute'):pipeline.prefill(torch.tensor([[1],[2]]),capacity=6)
            self.assertTrue(pipeline.cleanup_pending)
        pipeline.cleanup();self.assertFalse(pipeline.cleanup_pending)
        _,state=pipeline.prefill(torch.tensor([[1],[2]]),capacity=6);state.close()

    def test_configuration_and_shared_block_owners_are_rejected(self):
        import copy
        from damendaemon.glm_flash_layer_stage import FlashLayerStage
        from damendaemon.layer_pipeline import CpuLayerPipeline
        stages=list(self.pipeline().stages)
        with self.assertRaises(ValueError):CpuLayerPipeline(stages)
        stages[-1].config=copy.deepcopy(stages[-1].config)
        stages[-1].config['text_config']['rms_norm_eps']=2e-5
        with self.assertRaisesRegex(ValueError,'configurations'):FlashLayerPipeline(stages)
        # Different stage objects must not hide an overlapping prepared block.
        last=FlashLayerStage([stages[0].blocks[0],stages[1].blocks[-1]],start=1,total=3,
            norm=stages[1].norm,head=stages[1].head)
        with self.assertRaisesRegex(ValueError,'coverage'):FlashLayerPipeline([stages[0],last])
