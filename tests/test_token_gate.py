"""TokenGate property tests (#641): CPU, real-tensor convention.

Covers the issue's four property gates plus selection/bias/ratio unit
tests, against the synthetic hybrid fixture (4 layers: linear, full,
linear, full — exercising full-attention AND DeltaNet placements, with a
GDN layer above the gate processed over the kept sequence).
"""
import unittest
from unittest.mock import patch

import torch

from damendaemon.hybrid import HybridExecutor
from damendaemon.token_gate import RMS_EPS, TokenGate
from tests import test_hybrid_layer_range
from tests.test_xpu import ENV


class TokenGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_layer_range.HybridLayerRangeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.fixture.model()
        # The parent fixture doubles the layout to
        # [linear, full, linear, full]; gate layer 1 is full-attention with
        # layer 3 above (bias row) and layer 2 (GDN) above on the kept
        # sequence; gate layer 0 is a GDN placement (upstream's hybrid lane)
        # with bias rows at layers 1 and 3.
        assert self.model.config['layer_types'] == ['linear_attention', 'full_attention',
                                                   'linear_attention', 'full_attention']

    def executor(self, layer=1, keep=2, env=None):
        variables = {'DAMENDAEMON_TOKEN_GATE': '1',
                     'DAMENDAEMON_TOKEN_GATE_LAYER': str(layer),
                     'DAMENDAEMON_TOKEN_GATE_K': str(keep)}
        variables.update(env or {})
        with patch.dict('os.environ', variables), \
                patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(self.model, use_graph=False, max_batch=2,
                                   kernels='torch')

    def plain_executor(self):
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(self.model, use_graph=False, max_batch=2,
                                   kernels='torch')

    @staticmethod
    def _train_branch(gate, seed=7):
        """Give the gate a nonzero readout so it actually prunes."""
        generator = torch.Generator().manual_seed(seed)
        gate.branch_down = (torch.randn(gate.branch_down.shape, generator=generator)*.05)

    # ── property 1: zero scores reproduce standard prefill at any K ────────

    def test_zero_scores_reproduce_standard_prefill_byte_exactly_at_any_k(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6]
        reference = self.plain_executor().prefill(prompt)
        for keep in (0, 1, 3, 8):
            with self.subTest(keep=keep):
                gated = self.executor(keep=keep)
                result = gated.gated_prefill(prompt, 0, 2, 7, keep)
                self.assertTrue(torch.equal(result, reference))

    # ── property 2: K = |prunable| with zero scores is identity ─────────────

    def test_full_budget_with_zero_scores_is_identity(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11]
        reference = self.plain_executor().prefill(prompt)
        gated = self.executor(keep=len(prompt))
        result = gated.gated_prefill(prompt, 0, 1, 9, len(prompt))
        self.assertTrue(torch.equal(result, reference))
        for layer in gated.gate_valid:
            self.assertTrue(gated.gate_valid[layer][0, :len(prompt)].all())
        self.assertEqual(int(gated.gate_nvalid[0]), len(prompt))

    # ── property 3: non-prunable tokens always survive ──────────────────────

    def test_non_prunable_tokens_survive_every_budget(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11]
        span = (2, 9)
        non_article = [p for p in range(len(prompt)) if not span[0] <= p < span[1]]
        for keep in (0, 1, 2, 7):
            with self.subTest(keep=keep):
                gated = self.executor(keep=keep)
                self._train_branch(gated.gate)
                gated.gated_prefill(prompt, 0, *span, keep)
                valid = gated.gate_valid[3][0]
                kept = torch.nonzero(valid[:len(prompt)]).flatten().tolist()
                for position in non_article:
                    self.assertIn(position, kept)
                if keep == 0:
                    self.assertEqual(sorted(kept), sorted(non_article))
                # Full-attention layers above the gate write KV only at kept
                # positions; the gate layer itself keeps the full phase-1 KV.
                filled = gated.kv[3][0, 0, :, :len(prompt)].abs().sum((0, 2)) > 0
                self.assertEqual(filled.tolist(), valid[:len(prompt)].tolist())
                self.assertTrue(gated.kv[1][0, 0, :, :len(prompt)].abs().sum() > 0)

    def test_gated_prefill_decodes_through_hole_mask(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11, 12, 13]
        gated = self.executor(keep=1)
        self._train_branch(gated.gate)
        first = gated.gated_prefill(prompt, 0, 2, 11, 1).argmax(-1)[0]
        token = int(first)
        for _ in range(3):
            logits, next_ids = gated.decode(token)
            self.assertTrue(torch.isfinite(logits).all())
            token = int(next_ids[0])

    # ── property 4: chunked gated prefill equals whole gated prefill ───────

    def test_chunked_gated_equals_whole_gated(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11, 12, 13, 14, 15]
        span = (2, 13)
        for sizes in ((5, 5, 4), (14,), (2, 3, 4, 5), (7, 7)):
            with self.subTest(sizes=sizes):
                # A fresh whole-path reference per subtest: decode steps
                # mutate lengths/valid/nvalid, so a shared executor would
                # leak generated positions across subtests.
                whole = self.executor(keep=3)
                self._train_branch(whole.gate, seed=3)
                target = whole.gated_prefill(prompt, 0, *span, 3)
                kept_whole = torch.nonzero(whole.gate_valid[3][0]).flatten()
                chunked = self.executor(keep=3)
                self._train_branch(chunked.gate, seed=3)
                offset, actual = 0, None
                for index, size in enumerate(sizes):
                    final = index == len(sizes) - 1
                    actual = chunked.gated_prefill_chunk(
                        prompt[offset:offset+size], 0, offset, final, *span, 3)
                    if not final:
                        self.assertIsNone(actual)
                    offset += size
                torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
                kept_chunked = torch.nonzero(chunked.gate_valid[3][0]).flatten()
                self.assertTrue(torch.equal(kept_chunked, kept_whole))
                self.assertEqual(chunked.lengths[0], whole.lengths[0])
                self.assertEqual(int(chunked.gate_nvalid[0]), int(whole.gate_nvalid[0]))
                for layer in chunked.gate_valid:
                    torch.testing.assert_close(chunked.gate_fold[layer][0], whole.gate_fold[layer][0],
                                               rtol=2e-5, atol=2e-6)
                # The deferred phase-2 leaves identical recurrent state.
                for layer in whole.state:
                    torch.testing.assert_close(chunked.state[layer][0], whole.state[layer][0],
                                               rtol=2e-5, atol=2e-6)
                # Decode continuations agree.
                token = int(actual.argmax(-1)[0])
                for _ in range(3):
                    _, whole_next = whole.decode(token)
                    _, chunked_next = chunked.decode(token)
                    self.assertEqual(int(whole_next[0]), int(chunked_next[0]))
                    token = int(chunked_next[0])

    def test_chunked_zero_scores_fall_back_to_standard_prefill(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11]
        reference = self.plain_executor().prefill(prompt)
        chunked = self.executor(keep=2)
        offset, actual = 0, None
        for index, size in enumerate((4, 3, 3)):
            final = index == 2
            actual = chunked.gated_prefill_chunk(
                prompt[offset:offset+size], 0, offset, final, 2, 9, 2)
            offset += size
        self.assertTrue(torch.equal(actual, reference))

    # ── placements and admission ───────────────────────────────────────────

    def test_gdn_gate_placement_runs_with_bias_rows_on_full_attention_above(self):
        gated = self.executor(layer=0, keep=2)
        self.assertEqual(gated.gate.gate_layer, 0)
        self.assertEqual(gated.gate.bias_layers, [1, 3])
        self._train_branch(gated.gate, seed=5)
        logits = gated.gated_prefill([1, 4, 2, 7, 5, 3, 8, 6], 0, 1, 7, 2)
        self.assertTrue(torch.isfinite(logits).all())

    def test_gate_requires_full_attention_above(self):
        with self.assertRaisesRegex(ValueError, 'full-attention'):
            self.executor(layer=3)

    def test_default_off_leaves_executor_untouched(self):
        executor = self.plain_executor()
        self.assertIsNone(executor.gate)
        self.assertIsNone(executor.gate_valid)
        result = executor.prefill([1, 2, 3, 4])
        self.assertTrue(torch.isfinite(result).all())

    def test_gated_prefill_rejects_reuse_and_bad_spans(self):
        gated = self.executor(keep=2)
        self._train_branch(gated.gate)
        gated.gated_prefill([1, 2, 3, 4, 5, 6], 0, 1, 5, 2)
        with self.assertRaisesRegex(ValueError, 'cache slot'):
            gated.gated_prefill([7, 8, 9], 0, 0, 2, 2)
        fresh = self.executor(keep=2)
        with self.assertRaisesRegex(ValueError, 'span'):
            fresh.gated_prefill([1, 2, 3, 4], 0, 2, 2, 2)

    # ── gate math unit tests ────────────────────────────────────────────────

    def test_select_picks_highest_aggregate_scores(self):
        gate = TokenGate({'num_hidden_layers': 4, 'hidden_size': 8, 'intermediate_size': 16,
                         'layer_types': ['linear_attention', 'full_attention',
                                        'linear_attention', 'full_attention'],
                         'num_attention_heads': 2}, 1, [3], 2, 2)
        scores = torch.zeros(1, 6)
        scores[0, 2] = 0.9
        scores[0, 5] = 0.5
        scores[0, 4] = 0.7
        article = torch.tensor([False, True, True, False, True, True])
        kept = gate.select(scores, article, 2)
        self.assertEqual(kept.tolist(), [2, 4])
        # Budget covers everything: identity over the article span.
        self.assertEqual(gate.select(scores, article, 4).tolist(), [1, 2, 4, 5])
        # All-zero scores keep the whole span (untrained safety semantics).
        self.assertEqual(gate.select(torch.zeros(1, 6), article, 2).tolist(),
                         [1, 2, 4, 5])

    def test_bias_matches_reference_formula(self):
        gate = TokenGate({'num_hidden_layers': 4, 'hidden_size': 8, 'intermediate_size': 16,
                         'layer_types': ['linear_attention', 'full_attention',
                                        'linear_attention', 'full_attention'],
                         'num_attention_heads': 2}, 1, [3], 2, 2)
        gate.gate_scale = torch.tensor([[0.3, -0.4]])
        gate.norm_gain = torch.tensor([1.3])
        scores = torch.tensor([[0.0, 2.0, -1.0, 0.5]])
        article = torch.tensor([False, True, True, True])
        bias = gate.bias(scores, article)
        selected = scores[0][article]
        normalized = selected / torch.sqrt((selected.square().mean() + RMS_EPS)) * 1.3
        expected = normalized.unsqueeze(-1) * torch.nn.functional.softplus(torch.tensor([[0.3, -0.4]]))
        torch.testing.assert_close(bias[0, 1:, 0], expected.to(bias.dtype), rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.equal(bias[0, 0], torch.zeros_like(bias[0, 0])))

    def test_kept_ratio_identity_and_clamps(self):
        everything = torch.arange(8)
        self.assertTrue(torch.equal(TokenGate.kept_ratio(everything), torch.ones(8)))
        pruned = torch.tensor([0, 2, 5, 7])
        ratio = TokenGate.kept_ratio(pruned)
        self.assertEqual(float(ratio[0]), 1.0)
        self.assertTrue(bool((ratio <= 1.0).all()))
        self.assertTrue(bool((ratio[1:] > 0).all()))

    def test_decode_fold_is_head_mean_of_key_ratio_bias(self):
        gate = TokenGate({'num_hidden_layers': 4, 'hidden_size': 8, 'intermediate_size': 16,
                         'layer_types': ['linear_attention', 'full_attention',
                                        'linear_attention', 'full_attention'],
                         'num_attention_heads': 2}, 1, [3], 2, 2)
        bias = torch.arange(3*2*2, dtype=torch.float32).reshape(3, 2, 2)
        kept = torch.tensor([0, 1, 2])
        fold = gate.decode_fold(bias, kept)
        self.assertEqual(fold.shape, (3, 2))
        self.assertTrue(torch.allclose(fold, bias.mean(-1)))


if __name__ == '__main__':
    unittest.main()
