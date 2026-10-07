import unittest
from unittest.mock import patch

import torch

from dllm.expert_ffn import expert_feed_forward
from dllm.expert_selection import select_experts, select_token_experts, validate_token_expert_table
from dllm.floating_moe import floating_moe


class FloatingMoETests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        rng = torch.Generator().manual_seed(57)
        hidden = torch.randn(5, 4, generator=rng).to(dtype)
        banks = [torch.randn(shape, generator=rng).to(dtype) for shape in ((3, 6, 4), (3, 6, 4), (3, 4, 6))]
        shared = [torch.randn(shape, generator=rng).to(dtype) for shape in ((7, 4), (7, 4), (4, 7))]
        logits = torch.randn(5, 3, generator=rng)
        coefficients, selected = select_experts(logits, 2, scoring='sqrtsoftplus', renormalize=True, scale=1.5)
        return hidden, banks, shared, selected.contiguous(), coefficients.contiguous(), logits

    def test_composition_matches_individual_experts_and_bounded_chunks(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden, banks, shared, selected, coefficients, _ = self.fixture(dtype)
            expected = torch.zeros_like(hidden, dtype=torch.float32)
            for row in range(5):
                for route in range(2):
                    expert = selected[row, route].item()
                    expected[row] += expert_feed_forward(hidden[row:row+1], *(w[expert] for w in banks),
                                                         route_weights=coefficients[row:row+1, route:route+1], activation_limit=.7)[0].float()
            expected += expert_feed_forward(hidden, *shared, activation_limit=.7).float()
            for budget in (4096, 64*1024*1024):
                actual = floating_moe(hidden, *banks, selected, coefficients, shared=shared,
                                      activation_limit=.7, workspace_bytes=budget)
                torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-5 if dtype == torch.float32 else .02,
                                           atol=2e-6 if dtype == torch.float32 else .02)

    def test_zero_routes_leave_shared_contribution_unscaled(self):
        hidden, banks, shared, selected, coefficients, _ = self.fixture()
        actual = floating_moe(hidden, *banks, selected, torch.zeros_like(coefficients), shared=shared)
        torch.testing.assert_close(actual, expert_feed_forward(hidden, *shared))

    def test_token_table_routes_reorder_empty_and_no_host_extraction(self):
        hidden, banks, shared, _, _, logits = self.fixture()
        table = torch.tensor([[2, 0], [1, 2], [0, 1]])
        validate_token_expert_table(table, vocab_size=3, experts=3, top_k=2)
        coefficients, selected = select_token_experts(logits, torch.tensor([2, 0, 1, 2, 1]), table)
        coefficients, selected = coefficients.contiguous(), selected.contiguous()
        originals = [t.clone() for t in [hidden, *banks, *shared, selected, coefficients]]
        expected = floating_moe(hidden, *banks, selected, coefficients, shared=shared)
        order = torch.tensor([4, 0, 2, 1, 3])
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            actual = floating_moe(hidden[order], *banks, selected[order], coefficients[order], shared=shared)
            empty = floating_moe(hidden[:0], *banks, selected[:0], coefficients[:0], shared=shared)
        torch.testing.assert_close(actual, expected[order], rtol=2e-5, atol=1e-6)
        self.assertEqual(empty.shape, (0, 4))
        for tensor, original in zip([hidden, *banks, *shared, selected, coefficients], originals):
            torch.testing.assert_close(tensor, original)

    def test_workspace_limits_gather_rows(self):
        hidden, banks, _, selected, coefficients, _ = self.fixture()
        original = torch.bmm
        shapes = []
        def record(a, b):
            shapes.append(a.shape)
            return original(a, b)
        per_route = 3*6*4*4+64*6+32*4+128
        with patch('dllm.floating_moe.torch.bmm', side_effect=record):
            floating_moe(hidden, *banks, selected, coefficients, workspace_bytes=per_route*2)
        self.assertEqual(len(shapes), 15)
        self.assertTrue(all(shape[0] <= 2 for shape in shapes))

    def test_invalid_contracts_fail_before_computation(self):
        hidden, banks, shared, selected, coefficients, _ = self.fixture()
        with patch('dllm.floating_moe.torch.bmm', side_effect=AssertionError('computation started')):
            for options in ({'workspace_bytes':1}, {'workspace_bytes':True}, {'activation_limit':float('nan')},
                            {'shared':shared[:2]}, {'shared':[shared[0].half(), *shared[1:]]}):
                with self.assertRaises(ValueError):
                    floating_moe(hidden, *banks, selected, coefficients, **options)
            with self.assertRaises(ValueError):
                floating_moe(hidden, banks[0].half(), *banks[1:], selected, coefficients)
            with self.assertRaises(ValueError):
                floating_moe(hidden, *banks, selected, coefficients.half())
        invalid = selected.clone()
        invalid[0, 0] = -1
        with self.assertRaises((IndexError, RuntimeError)):
            floating_moe(hidden, *banks, invalid, coefficients)
