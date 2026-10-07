from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_layer_pipeline import GlmLayerPipeline
from dllm.glm_reference_loader import load_reference_stage
from tests.test_glm_layer_stage import prepared
from tests.test_glm_model import model_fixture
from tests.test_glm_reference_loader import fixture, floating_control


class GlmLayerPipelineTests(unittest.TestCase):
    def pipeline(self): return GlmLayerPipeline(prepared(model_fixture(),[(0,2),(2,3)]))

    def test_packed_pipeline_logits_cursors_reorder_and_fresh_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,_,values=fixture(path)
            pipeline=GlmLayerPipeline([load_reference_stage(path,0,2),load_reference_stage(path,2,3)])
            tokens=torch.tensor([[1,4,3,2],[2,3,1,5]])
            control=floating_control(config,values)
            expected,other=control.prefill(tokens,capacity=5,all_logits=True);self.addCleanup(other.close)
            first,state=pipeline.prefill(tokens[:,:2],capacity=5,all_logits=True);self.addCleanup(state.close)
            torch.testing.assert_close(first,expected[:,:2],rtol=3e-5,atol=3e-6)
            order=torch.tensor([1,0,1]);state.reorder(order)
            actual=pipeline.forward(tokens[order,2:3],state,position=2)
            torch.testing.assert_close(actual,expected[order,2],rtol=3e-5,atol=3e-6)
            state.reorder(torch.tensor([2]));actual=pipeline.forward(tokens[1:2,3:4],state,position=3)
            torch.testing.assert_close(actual,expected[1:2,3],rtol=3e-5,atol=3e-6)
            self.assertEqual(state.position,4)
            self.assertTrue(all(local.position==4 for local in state.stages))
            state.close();state.close()
            fresh,next_state=pipeline.prefill(tokens,capacity=5,all_logits=True);next_state.close()
            torch.testing.assert_close(fresh,expected,rtol=3e-5,atol=3e-6)

    def test_composition_and_aggregate_admission_precede_allocation(self):
        stages=prepared(model_fixture(),[(0,2),(2,3)])
        for wrong in ([stages[0]],stages[::-1],[stages[0],stages[0]], [stages[1],stages[1]]):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):GlmLayerPipeline(wrong)
        pipeline=GlmLayerPipeline(stages)
        exact=2*pipeline.cache_bytes_per_row(4)
        with patch.object(stages[0],'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'aggregate'):pipeline.new_state(2,4,max_cache_bytes=exact-1)
        state=pipeline.new_state(2,4,max_cache_bytes=exact);self.addCleanup(state.close)
        with self.assertRaisesRegex(ValueError,'aggregate'):state.reorder(torch.tensor([1,0,1]))
        self.assertEqual(state.batch,2);self.assertFalse(state.poisoned)
        # Preflight the final output allowance before even the first stage runs.
        stages[-1].max_output_bytes=1
        with patch.object(stages[0],'forward',side_effect=AssertionError('executed')):
            with self.assertRaisesRegex(ValueError,'output budget'):pipeline.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)

    def test_later_execution_and_reorder_failures_poison_entire_epoch(self):
        pipeline=self.pipeline();state=pipeline.new_state(2,4);self.addCleanup(state.close)
        with patch.object(pipeline.stages[-1],'forward',side_effect=RuntimeError('later stage')):
            with self.assertRaisesRegex(RuntimeError,'later stage'):pipeline.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertEqual(state.position,0);self.assertEqual(state.stages[0].position,1)
        self.assertTrue(state.poisoned);self.assertTrue(all(local.poisoned for local in state.stages))
        with self.assertRaises(RuntimeError):pipeline.forward(torch.tensor([[1],[2]]),state,position=0)
        state.close()
        _,state=pipeline.prefill(torch.tensor([[1],[2]]),capacity=4);self.addCleanup(state.close)
        with patch.object(state.stages[-1],'reorder',side_effect=MemoryError('later reorder')):
            with self.assertRaises(MemoryError):state.reorder(torch.tensor([1]))
        self.assertTrue(all(local.poisoned for local in state.stages))
        with self.assertRaises(RuntimeError):state.reorder(torch.tensor([0]))

    def test_close_attempts_all_stages_and_retains_only_failed_cleanup(self):
        pipeline=self.pipeline();state=pipeline.new_state(1,4)
        first,last=state.stages
        with patch.object(first,'close',side_effect=RuntimeError('first close')):
            with self.assertRaisesRegex(RuntimeError,'first close'):state.close()
        self.assertTrue(last.closed);self.assertEqual(state.stages,[first])
        self.assertIs(state.owner,pipeline)
        state.close();self.assertIsNone(state.owner);self.assertFalse(state.stages)

    def test_failed_prefill_cleanup_blocks_allocation_until_retry_succeeds(self):
        pipeline=self.pipeline();state=pipeline.new_state(1,4)
        with patch.object(pipeline,'new_state',return_value=state),patch.object(pipeline.stages[-1],'forward',side_effect=RuntimeError('primary')),\
                patch.object(state.stages[0],'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary'):pipeline.prefill(torch.tensor([[1]]),capacity=4)
        self.assertEqual(pipeline.failed_cleanups,[state]);self.assertTrue(state.closed)
        pending=state.stages[0]
        with patch.object(pending,'close',side_effect=RuntimeError('still cleanup')),patch.object(pipeline.stages[0],'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(RuntimeError,'still cleanup'):pipeline.new_state(1,4)
        pipeline.cleanup();self.assertFalse(pipeline.failed_cleanups)
        _,fresh=pipeline.prefill(torch.tensor([[1]]),capacity=4);fresh.close()

    def test_allocation_failure_cleans_earlier_stage_and_retains_later_owner(self):
        pipeline=self.pipeline();first=pipeline.stages[0].new_state(1,4)
        last=pipeline.stages[-1];local=last.new_state(1,4)
        def fail(*args,**kwargs):
            last.failed_cleanups.append(local);raise MemoryError('allocation')
        with patch.object(pipeline.stages[0],'new_state',return_value=first),patch.object(last,'new_state',fail),patch.object(local,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(MemoryError,'allocation'):pipeline.new_state(1,4)
        self.assertTrue(first.closed);self.assertEqual(last.failed_cleanups,[local])
        pipeline.cleanup();self.assertFalse(last.failed_cleanups);self.assertTrue(local.closed)

    def test_invalid_inputs_do_not_mutate_but_external_cursor_drift_poisons_all(self):
        pipeline=self.pipeline();state=pipeline.new_state(1,4);self.addCleanup(state.close)
        for tokens,position in ((torch.tensor([[99]]),0),(torch.tensor([[1]]),1),(torch.tensor([[1],[2]]),0)):
            with self.assertRaises(ValueError):pipeline.forward(tokens,state,position=position)
        self.assertFalse(state.poisoned);self.assertEqual(state.position,0)
        state.stages[-1].position=1
        with self.assertRaises(RuntimeError):pipeline.forward(torch.tensor([[1]]),state,position=0)
        self.assertTrue(state.poisoned);self.assertTrue(all(local.poisoned for local in state.stages))
