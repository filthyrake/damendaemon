"""Device-resident expert routing must match the CPU selection semantics.

Equivalence runs on the same device (CPU) so the two implementations are
directly comparable; the A2 fixture separately proves CUDA-tensor operation.
The device tie contract (lower expert ID wins) is tested explicitly, since the
CPU plain-topk tie order is unspecified.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from damendaemon.cuda_expert_routing import select_experts_device
from damendaemon.expert_selection import select_experts


def fixture(tokens=6, experts=8, seed=5, ties=False):
    rng = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=rng)
    if ties:
        logits[:, 1] = logits[:, 4]  # deliberate cross-column ties
        logits[0, :] = logits[0, 2]  # one fully tied row
    return logits


class DeviceRoutingEquivalenceTests(unittest.TestCase):
    def test_matches_cpu_reference_across_policies(self):
        logits = fixture()
        cases = [
            dict(),
            dict(renormalize=True),
            dict(scoring='sigmoid', renormalize=True),
            dict(scoring='sqrtsoftplus'),
            dict(scale=3),
            dict(renormalize=True, scale=2),
            dict(group_count=2, groups_to_keep=1),
            dict(group_count=2, groups_to_keep=2, renormalize=True),
            dict(correction_bias=torch.linspace(-.2, .2, 8)),
            dict(group_count=2, correction_bias=torch.linspace(-.2, .2, 8), renormalize=True),
            dict(group_count=4, groups_to_keep=2, top_k=3),
        ]
        for case in cases:
            top_k = case.pop('top_k', 2)
            with self.subTest(case=case):
                expected_w, expected_s = select_experts(logits, top_k, **case)
                actual_w, actual_s = select_experts_device(logits, top_k, **case)
                torch.testing.assert_close(actual_w, expected_w)
                self.assertTrue(torch.equal(actual_s, expected_s))

    def test_grouped_bias_ties_prefer_lower_ids_like_cpu(self):
        logits = fixture(ties=True)
        bias = torch.zeros(8)
        for top_k in (1, 2, 4):
            with self.subTest(top_k=top_k):
                expected_w, expected_s = select_experts(logits, top_k, group_count=2,
                                                        correction_bias=bias)
                actual_w, actual_s = select_experts_device(logits, top_k, group_count=2,
                                                           correction_bias=bias)
                torch.testing.assert_close(actual_w, expected_w)
                self.assertTrue(torch.equal(actual_s, expected_s))

    def test_device_tie_contract_prefers_lower_ids_on_plain_topk(self):
        logits = fixture(ties=True)
        weights, selected = select_experts_device(logits, 3)
        # Row 0 is fully tied: IDs must come out 0,1,2 with equal weights.
        self.assertTrue(torch.equal(selected[0], torch.tensor([0, 1, 2])))
        torch.testing.assert_close(weights[0], torch.full((3,), logits[0].softmax(-1)[0]))
        # Tied columns 1 and 4: lower ID first wherever both fit in top-k.
        self.assertLess(selected[:, 1].min(), 4)

    def test_preselected_routes_match_cpu_reference_and_keep_order(self):
        logits = fixture()
        routes = torch.tensor([[7, 0], [3, 3], [5, 2], [1, 6], [0, 7], [2, 4]])
        for scoring in ('softmax', 'sigmoid', 'sqrtsoftplus'):
            with self.subTest(scoring=scoring):
                expected_w, expected_s = select_experts(logits, 2, scoring=scoring,
                                                        renormalize=True, selected_experts=routes)
                actual_w, actual_s = select_experts_device(logits, 2, scoring=scoring,
                                                           renormalize=True, selected_experts=routes)
                torch.testing.assert_close(actual_w, expected_w)
                self.assertTrue(torch.equal(actual_s, expected_s))
        with self.assertRaisesRegex(ValueError, 'without grouping or bias'):
            select_experts_device(logits, 2, selected_experts=routes, group_count=2)
        with self.assertRaisesRegex(ValueError, 'without grouping or bias'):
            select_experts_device(logits, 2, selected_experts=routes, correction_bias=torch.zeros(8))

    def test_no_host_readback_inside_the_call(self):
        logits = fixture()
        with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')), \
                patch('torch.Tensor.tolist', side_effect=AssertionError('host readback')):
            for case in ({}, dict(group_count=2, correction_bias=torch.zeros(8))):
                with self.subTest(case=case):
                    weights, selected = select_experts_device(logits, 2, **case)
        self.assertEqual(weights.device, logits.device)
        self.assertEqual(selected.dtype, torch.int64)

    def test_budget_geometry_and_input_rejection(self):
        logits = fixture()
        with self.assertRaisesRegex(ValueError, 'workspace budget'):
            select_experts_device(logits, 2, max_workspace_bytes=64)
        with self.assertRaisesRegex(ValueError, 'invalid expert selection policy'):
            select_experts_device(logits, 0)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite expert routing logits'):
            select_experts_device(torch.tensor([[float('nan'), 0., 0., 0., 0., 0., 0., 0.]]), 2)
        with self.assertRaisesRegex(ValueError, 'correction bias'):
            select_experts_device(logits, 2, correction_bias=torch.zeros(3))

    def test_empty_batch_routes_without_host_readback(self):
        logits = torch.zeros(0, 8)
        with patch('torch.Tensor.item', side_effect=AssertionError('host readback')):
            for case in ({}, dict(group_count=2, correction_bias=torch.zeros(8))):
                with self.subTest(case=case):
                    weights, selected = select_experts_device(logits, 2, **case)
                    self.assertEqual(weights.shape, (0, 2))
                    self.assertEqual(selected.shape, (0, 2))

    def test_device_routes_feed_prepared_experts_directly(self):
        # The routing outputs must satisfy the packed-MoE admission contract
        # (int64 device IDs, matching FP32 coefficients) without reshaping.
        from damendaemon.cuda_packed_moe import _accumulate
        logits = fixture(tokens=4)
        for case in ({}, dict(group_count=2, correction_bias=torch.zeros(8), renormalize=True)):
            with self.subTest(case=case):
                weights, selected = select_experts_device(logits, 2, **case)
                hidden = torch.randn(4, 2)
                bank = [SimpleNamespace(device=torch.device('cpu'), width=2,
                                        _row=lambda x, c, m=m: (x.float()*m).to(x.dtype))
                        for m in (2.+m/8 for m in range(8))]
                routed = _accumulate(hidden, bank, selected, weights, bank[0])
                host_w, host_s = select_experts(logits, 2, **case)
                expected = _accumulate(hidden, bank, host_s, host_w, bank[0])
                torch.testing.assert_close(routed, expected)


if __name__ == '__main__':
    unittest.main()
