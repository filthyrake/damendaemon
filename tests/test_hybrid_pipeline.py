"""CPU coordinator/state tests; no GPU transfer, network or performance proof."""
import unittest
from unittest.mock import patch

import torch

from dllm.hybrid_pipeline import HybridPipeline, PipelineCancelled
from tests import test_hybrid_stage


class HybridPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.stages = [self.fixture.stage(0, 2), self.fixture.stage(2, 4)]
        self.pipeline = HybridPipeline(self.stages)

    def test_prefill_chunks_reordered_decode_and_slot_reuse(self):
        full = self.fixture.fixture.fixture.executor()
        self.assertIsNone(self.pipeline.prefill_chunk([1, 2], 0, 0, False))
        torch.testing.assert_close(self.pipeline.prefill_chunk([3, 4], 0, 2), full.prefill([1, 2, 3, 4]),
                                   rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(self.pipeline.prefill([5, 6], 1), full.prefill([5, 6], 1), rtol=0, atol=0)
        for tokens, slots in (([7, 8], [1, 0]), ([9], [0]), ([10], [1])):
            torch.testing.assert_close(self.pipeline.decode_batch(tokens, slots), full.decode_batch(tokens, slots),
                                       rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(self.pipeline.prefill([11, 12], 0), full.prefill([11, 12], 0), rtol=0, atol=0)
        self.assertEqual(self.pipeline.lengths, full.lengths)
        for stage in self.stages:
            self.assertEqual(stage.lengths, full.lengths)
            for name in ('kv', 'conv', 'state'):
                for layer, tensor in getattr(stage, name).items():
                    torch.testing.assert_close(tensor, getattr(full, name)[layer], rtol=2e-5, atol=2e-6)

    def test_late_stage_failure_poisoning_prevents_cursor_commit_or_reuse(self):
        with patch.object(self.stages[1], 'forward_layers', side_effect=RuntimeError('injected stage failure')):
            with self.assertRaises(RuntimeError):
                self.pipeline.prefill([1, 2])
        self.assertTrue(self.pipeline.failed)
        self.assertEqual(self.pipeline.lengths, [0, 0])
        self.assertTrue(all(stage.lengths == [0, 0] for stage in self.stages))
        with self.assertRaises(RuntimeError):
            self.pipeline.prefill([1, 2])
        with self.assertRaises(ValueError):
            HybridPipeline(self.stages)
        self.pipeline.close()
        for stage in self.stages:
            self.assertTrue(stage.closed)
            self.assertFalse(stage.w or stage.kv or stage.state or stage.conv or stage.model.weights)
        self.assertTrue(self.fixture.model.weights)  # Host source owner is separate.
        status = self.pipeline.status()
        self.assertTrue(status['closed'])
        self.assertTrue(all(s['resident_weight_bytes'] == s['state_bytes'] == 0 for s in status['stages']))

    def test_device_completion_failure_is_not_a_successful_commit(self):
        with patch.object(self.stages[1], 'synchronize', side_effect=RuntimeError('injected sync failure')):
            with self.assertRaises(RuntimeError):
                self.pipeline.prefill([1, 2])
            with self.assertRaises(RuntimeError):
                self.pipeline.close()
            self.assertFalse(self.pipeline.closed)
            self.assertTrue(all(stage.w for stage in self.stages))
        self.assertEqual(self.pipeline.lengths, [0, 0])
        self.pipeline.close()
        self.pipeline.close()  # Idempotent after actual success.

    def test_cancellation_before_work_keeps_pipeline_and_midstep_poisons_it(self):
        with self.assertRaises(PipelineCancelled):
            self.pipeline.prefill([1, 2], cancelled=lambda: True)
        self.assertFalse(self.pipeline.failed)
        checks = 0
        def cancelled():
            nonlocal checks
            checks += 1
            return checks == 3  # Before the second stage, after first-stage mutation.
        with self.assertRaises(PipelineCancelled):
            self.pipeline.prefill([1, 2], cancelled=cancelled)
        self.assertTrue(self.pipeline.failed)
        self.assertEqual(self.pipeline.lengths, [0, 0])

    def test_invalid_inputs_do_not_poison_or_mutate(self):
        for call in (lambda: self.pipeline.prefill([-1]), lambda: self.pipeline.prefill([1], True),
                     lambda: self.pipeline.prefill_chunk([1], 0, 2),
                     lambda: self.pipeline.decode_batch([1], [0]),
                     lambda: self.pipeline.prefill_chunk([1], 0, 0, 1)):
            with self.assertRaises(ValueError):
                call()
        self.assertFalse(self.pipeline.failed)
        self.assertEqual(self.pipeline.lengths, [0, 0])

    def test_missing_ranges_and_foreign_models_rejected(self):
        with self.assertRaises(ValueError):
            HybridPipeline([self.fixture.stage(0, 1), self.fixture.stage(2, 4)])
        with self.assertRaises(ValueError):
            HybridPipeline([self.fixture.stage(0, 2)])
        left, right = self.fixture.stage(0, 2), self.fixture.stage(2, 4)
        right.source_identity = object()
        with self.assertRaises(ValueError):
            HybridPipeline([left, right])
