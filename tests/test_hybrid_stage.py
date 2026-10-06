"""Stage-local CPU allocation and composition tests; no multi-GPU qualification."""
import unittest
from unittest.mock import patch

import torch

from dllm.capacity import hybrid_state_bytes
from dllm.hybrid_stage import HybridLayerStage, shard_model
from tests import test_hybrid_layer_range


class HybridStageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_layer_range.HybridLayerRangeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.fixture.model()

    def stage(self, start, end):
        with patch('dllm.hybrid.selected_device', return_value='cpu'), \
                patch('dllm.hybrid.empty_cache'), patch('dllm.hybrid.available_memory', return_value=2**40):
            return HybridLayerStage(self.model, start, end, max_batch=2)

    def test_shards_own_only_selected_weights_and_keep_global_indices(self):
        left, right = shard_model(self.model, 0, 2), shard_model(self.model, 2, 4)
        self.assertIn('embed', left.weights)
        self.assertNotIn('head', left.weights)
        self.assertNotIn('norm', left.weights)
        self.assertNotIn('embed', right.weights)
        self.assertIn('head', right.weights)
        self.assertIn('2.qkv', right.weights)
        self.assertNotIn('0.qkv', right.weights)
        self.assertIs(left.weights['embed'], self.model.weights['embed'])
        self.assertIs(right.weights['head'], self.model.weights['head'])
        self.assertEqual(right.config['num_hidden_layers'], 4)
        # Tied embedding/head will reside separately on different stages.
        embed_bytes = self.model.weights['embed'].numel()*self.model.weights['embed'].element_size()
        self.assertEqual(left.bytes+right.bytes, self.model.bytes+embed_bytes)
        self.assertIn('head', self.model.weights)  # Source model remains intact.

    def test_allocation_budget_counts_only_owned_layer_state(self):
        for start, end in ((0, 1), (1, 3), (3, 4)):
            with self.subTest(start=start, end=end):
                stage = self.stage(start, end)
                owned = set(range(start, end))
                self.assertEqual(set(stage.kv) | set(stage.state), owned)
                self.assertEqual(set(stage.conv), set(stage.state))
                self.assertEqual(set(stage.w), set(stage.model.weights))
                state_config = {**self.model.config, 'num_hidden_layers': end-start,
                                'layer_types': self.model.config['layer_types'][start:end]}
                state_bytes = hybrid_state_bytes(state_config, 32, 2, 4)
                actual = sum(t.numel()*t.element_size() for group in (stage.kv, stage.state, stage.conv)
                             for t in group.values())
                self.assertEqual(actual, state_bytes)
                self.assertEqual(stage.required_memory_bytes, stage.model.bytes+state_bytes+512*1024**2)
                with self.assertRaises(ValueError):
                    stage.forward_layers(None, None, None, True, layer_start=0, layer_end=4)

    def test_disjoint_stages_compose_prefill_chunks_and_decode(self):
        full = self.fixture.fixture.executor()
        left, right = self.stage(0, 2), self.stage(2, 4)
        def run(tokens, start=0, prefill=True):
            positions = torch.arange(start, start+len(tokens))[None]
            slots = torch.tensor([0])
            hidden = left.forward_layers(left.embedding(torch.tensor([tokens])), positions, slots,
                                         prefill, start if prefill else 0)
            hidden = right.forward_layers(hidden, positions, slots, prefill, start if prefill else 0)
            return right.output_logits(hidden[:, -1])
        run([1, 2])
        torch.testing.assert_close(run([3, 4, 5], 2), full.prefill([1, 2, 3, 4, 5]), rtol=2e-5, atol=2e-6)
        for start, token in enumerate((6, 7, 8), 5):
            torch.testing.assert_close(run([token], start, False), full.decode(token)[0], rtol=2e-5, atol=2e-6)
        for name in ('kv', 'conv', 'state'):
            for layer, tensor in getattr(full, name).items():
                stage = left if layer < 2 else right
                torch.testing.assert_close(getattr(stage, name)[layer], tensor, rtol=2e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, 'coordinator'):
            left.prefill([1, 2])
        with self.assertRaisesRegex(ValueError, 'first'):
            right.embedding(torch.tensor([[1]]))
        with self.assertRaisesRegex(ValueError, 'last'):
            left.output_logits(torch.ones(1, self.model.config['hidden_size']))

    def test_invalid_ranges_and_quantization_fail_before_allocation(self):
        for start, end in ((True, 2), (-1, 2), (1, 1), (0, 5)):
            with self.assertRaises(ValueError):
                shard_model(self.model, start, end)
        self.model.quantization = 'q8_0'
        with self.assertRaises(ValueError):
            self.stage(0, 2)
