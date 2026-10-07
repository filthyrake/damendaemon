import math
import unittest
from unittest.mock import patch

import torch

from dllm.expert_ffn import expert_feed_forward


class ExpertFeedForwardTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        rng = torch.Generator().manual_seed(31)
        return tuple(torch.randn(shape, generator=rng).to(dtype)
                     for shape in ((3, 4), (6, 4), (6, 4), (4, 6)))

    def test_independent_scalar_arithmetic_for_shared_and_routed_experts(self):
        hidden, gate, up, down = self.fixture()
        coefficients = torch.tensor([[.2], [.7], [1.3]])
        for limit in (None, .5):
            for route in (None, coefficients):
                expected = []
                for index, row in enumerate(hidden.tolist()):
                    gate_values = [sum(x*w for x, w in zip(row, weights)) for weights in gate.tolist()]
                    up_values = [sum(x*w for x, w in zip(row, weights)) for weights in up.tolist()]
                    if limit is not None:
                        gate_values = [min(value, limit) for value in gate_values]
                        up_values = [max(-limit, min(value, limit)) for value in up_values]
                    activated = [g/(1+math.exp(-g))*u for g, u in zip(gate_values, up_values)]
                    if route is not None:
                        activated = [value*route[index, 0].item() for value in activated]
                    expected.append([sum(x*w for x, w in zip(activated, weights)) for weights in down.tolist()])
                actual = expert_feed_forward(hidden, gate, up, down,
                                             route_weights=route, activation_limit=limit)
                torch.testing.assert_close(actual, torch.tensor(expected), rtol=2e-5, atol=1e-6)

    def test_low_precision_weighting_occurs_before_down_projection_cast(self):
        for dtype in (torch.float16, torch.bfloat16):
            hidden, gate, up, down = self.fixture(dtype)
            route = torch.tensor([[.21], [.73], [1.31]])
            gate_values = torch.nn.functional.linear(hidden, gate).float().clamp(max=1.)
            up_values = torch.nn.functional.linear(hidden, up).float().clamp(-1., 1.)
            activated = torch.nn.functional.silu(gate_values)*up_values
            expected = torch.nn.functional.linear((activated*route).to(dtype), down)
            actual = expert_feed_forward(hidden, gate, up, down, route_weights=route, activation_limit=1.)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            late_weighted = (torch.nn.functional.linear(activated.to(dtype), down).float()*route).to(dtype)
            self.assertFalse(torch.equal(actual, late_weighted))
            self.assertEqual(actual.dtype, dtype)

    def test_negative_gate_is_not_clamped_up_and_shared_output_is_unscaled(self):
        hidden = torch.tensor([[1.]])
        gate, up, down = torch.tensor([[-20.]]), torch.tensor([[20.]]), torch.tensor([[1.]])
        result = expert_feed_forward(hidden, gate, up, down, activation_limit=10.)
        self.assertLess(abs(result.item()), 1e-6)
        shared = expert_feed_forward(*self.fixture())
        zero_routed = expert_feed_forward(*self.fixture(), route_weights=torch.zeros(3, 1))
        torch.testing.assert_close(shared+zero_routed, shared, rtol=0, atol=0)

    def test_reordered_and_empty_rows_without_host_tensor_extraction(self):
        hidden, gate, up, down = self.fixture()
        expected = expert_feed_forward(hidden, gate, up, down)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            actual = expert_feed_forward(hidden[[2, 0, 1]], gate, up, down)
            empty = expert_feed_forward(hidden[:0], gate, up, down)
        torch.testing.assert_close(actual, expected[[2, 0, 1]], rtol=2e-5, atol=1e-6)
        self.assertEqual(empty.shape, (0, 4))

    def test_incompatible_weights_and_options_fail_before_projection(self):
        hidden, gate, up, down = self.fixture()
        invalid = [(hidden, gate, up[:5], down), (hidden, gate.half(), up, down),
                   (hidden, gate, up, down[:, :5]), (hidden.to(torch.int32), gate, up, down)]
        with patch('dllm.expert_ffn.F.linear', side_effect=AssertionError('projection started')):
            for tensors in invalid:
                with self.assertRaises(ValueError):
                    expert_feed_forward(*tensors)
            for options in ({'route_weights':torch.ones(3)}, {'route_weights':torch.ones(3, 1).half()},
                            {'activation_limit':True}, {'activation_limit':float('nan')}, {'activation_limit':-1}):
                with self.assertRaises(ValueError):
                    expert_feed_forward(hidden, gate, up, down, **options)
