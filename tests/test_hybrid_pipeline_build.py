"""CPU construction/failure fixtures, not CUDA qualification."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from dllm.hybrid_pipeline_build import build_cuda_pipeline, PipelineBuildError
from dllm.hybrid_stage import HybridLayerStage
from tests import test_hybrid_stage


class PipelineBuildTests(unittest.TestCase):
    def setUp(self):
        fixture = test_hybrid_stage.HybridStageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.model = fixture.model
        self.model.weights = {k: v.half() if v.ndim > 1 else v.float()
                              for k, v in self.model.weights.items()}
        self.model.weights['head'] = self.model.weights['embed']
        self.patches = {}
        replacements = {
            'dllm.hybrid.selected_device': {'return_value': 'cpu'},
            'dllm.hybrid.empty_cache': {},
            'dllm.hybrid.available_memory': {'return_value': 2**40},
            'torch.cuda.is_available': {'return_value': True},
            'torch.cuda.device_count': {'return_value': 2},
            'torch.cuda.mem_get_info': {'return_value': (2**40, 2**40)},
            'torch.cuda.device': {'side_effect': lambda device: nullcontext()},
            'torch.cuda.synchronize': {},
            'torch.cuda.empty_cache': {},
        }
        for name, kwargs in replacements.items():
            replacement = patch(name, **kwargs)
            self.patches[name] = replacement.start()
            self.addCleanup(replacement.stop)
        env = patch.dict('os.environ', {'DLLM_DEVICE': 'cuda',
             'DLLM_EXPERIMENTAL_HYBRID_PIPELINE': '1'}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def build(self):
        return build_cuda_pipeline(self.model, [0, 1], [2**30, 2**30], max_batch=2)

    def test_fitting_construction_executes_and_preserves_source(self):
        pipeline = self.build()
        self.assertEqual(pipeline.devices, (0, 1))
        self.assertEqual(len(pipeline.stages), 2)
        self.assertEqual(pipeline.stages[0].layer_start, 0)
        self.assertEqual(pipeline.stages[-1].layer_end, 4)
        logits = pipeline.prefill([1, 2, 3])
        self.assertEqual(tuple(logits.shape), (1, self.model.config['vocab_size']))
        self.assertTrue(torch.isfinite(logits).all())
        control = build_cuda_pipeline(self.model, [0], [2**30], max_batch=2)
        torch.testing.assert_close(logits, control.prefill([1, 2, 3]), rtol=0, atol=0)
        torch.testing.assert_close(pipeline.decode_batch([4], [0]),
                                   control.decode_batch([4], [0]), rtol=0, atol=0)
        control.close()
        for stage, budget in zip(pipeline.stages, pipeline.placement['stages']):
            self.assertEqual(stage.required_memory_bytes, budget['required_memory_bytes'])
            self.assertLessEqual(stage.required_memory_bytes, 2**30)
        pipeline.close()
        self.assertTrue(self.model.weights)

    def test_invalid_and_no_fit_fail_before_device_work(self):
        for devices, budgets in (([0, 0], [2**30]*2), ([True], [2**30]),
                                 ([0, 1], [2**30]), ([0, 1], [1, 1])):
            with self.subTest(devices=devices, budgets=budgets), self.assertRaises((ValueError, MemoryError)):
                build_cuda_pipeline(self.model, devices, budgets)
        self.patches['torch.cuda.mem_get_info'].assert_not_called()
        self.patches['torch.cuda.device'].assert_not_called()
        with patch.dict('os.environ', DLLM_EXPERIMENTAL_HYBRID_PIPELINE='0'), self.assertRaises(ValueError):
            self.build()

    def test_observed_memory_can_reject_before_stage_construction(self):
        self.patches['torch.cuda.mem_get_info'].return_value = (1, 2**40)
        with patch.object(HybridLayerStage, '__init__') as constructor, self.assertRaises(MemoryError):
            self.build()
        constructor.assert_not_called()

    def test_partial_failure_cleans_both_completed_and_partial_stages(self):
        original = HybridLayerStage.__init__
        created = []
        def fail(stage, *args, **kwargs):
            original(stage, *args, **kwargs)
            created.append(stage)
            if len(created) == 2:
                raise MemoryError('injected after allocation')
        with patch.object(HybridLayerStage, '__init__', fail), self.assertRaises(PipelineBuildError) as result:
            self.build()
        self.assertTrue(result.exception.closed)
        self.assertTrue(all(s.closed and not (s.w or s.kv or s.state or s.conv) for s in created))
        self.assertTrue(self.model.weights)
        result.exception.close()

    def test_failed_cleanup_retains_retry_ownership(self):
        original = HybridLayerStage.__init__
        def fail(stage, *args, **kwargs):
            original(stage, *args, **kwargs)
            raise MemoryError('injected allocation failure')
        sync = self.patches['torch.cuda.synchronize']
        sync.side_effect = RuntimeError('injected device failure')
        with patch.object(HybridLayerStage, '__init__', fail), self.assertRaises(PipelineBuildError) as result:
            self.build()
        failure = result.exception
        self.assertFalse(failure.closed)
        self.assertTrue(failure.stages[0].w)
        self.assertTrue(any('cleanup incomplete' in note for note in failure.__notes__))
        sync.side_effect = None
        failure.close()
        self.assertTrue(failure.closed)
        self.assertFalse(failure.stages[0].w)

    def test_changed_headroom_cleans_prior_stage(self):
        self.patches['torch.cuda.mem_get_info'].side_effect = [
            (2**40, 2**40), (2**40, 2**40), (2**40, 2**40), (1, 2**40)]
        with self.assertRaises(PipelineBuildError) as result:
            self.build()
        self.assertTrue(result.exception.closed)
        self.assertEqual(len(result.exception.stages), 1)
        self.assertTrue(result.exception.stages[0].closed)

    def test_interrupt_before_stage_fields_exist_still_cleans(self):
        with patch.object(HybridLayerStage, '__init__', side_effect=KeyboardInterrupt), \
                self.assertRaises(PipelineBuildError) as result:
            self.build()
        self.assertTrue(result.exception.closed)
        self.assertTrue(result.exception.stages[0].closed)
        self.assertIn('KeyboardInterrupt', str(result.exception))
