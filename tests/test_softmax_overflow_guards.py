"""Overflowing softmax inputs must fail instead of silently producing NaN.

Covers the ROCm grouped prefill, the GPT-OSS router, and the Qwen3.5 vision
eager attention — the same silent-corruption class as #477.
"""
import unittest
from types import SimpleNamespace

import torch

from dllm.grouped_attention import attention
from dllm.gpt_oss import GptOssExecutor
from dllm.vision_qwen35 import _eager_attention


class GroupedAttentionOverflowTests(unittest.TestCase):
    def test_overflowing_scores_fail_and_ordinary_output_matches_reference(self):
        torch.manual_seed(7)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mask = torch.ones(4, 4, dtype=torch.bool).tril()
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        torch.testing.assert_close(attention(q, k, v), expected, atol=1e-5, rtol=1e-5)
        hot_q = q.clone(); hot_q[0, 0, 0, 0] = 1e30
        hot_k = k.clone(); hot_k[0, 0, 0, 0] = 1e30
        with self.assertRaisesRegex(RuntimeError, 'nonfinite grouped attention scores'):
            attention(hot_q, hot_k, v)

    def test_nonfinite_values_fail_at_publication(self):
        q = torch.zeros(1, 2, 4, 8)
        k = torch.zeros(1, 2, 6, 8)
        v = torch.zeros(1, 2, 6, 8); v[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'nonfinite grouped attention output'):
            attention(q, k, v)


class GptOssRoutingOverflowTests(unittest.TestCase):
    def _fake_executor(self, router_output, captured):
        return SimpleNamespace(
            w={'0.router': object(), '0.router.bias': None,
               '0.up.blocks': object(), '0.up.scales': object(), '0.up.bias': None,
               '0.down.blocks': object(), '0.down.scales': object(), '0.down.bias': None},
            c={'num_experts_per_tok': 2, 'swiglu_limit': 7.0}, grouped_prefill=None,
            chunked_experts=False, normal_experts=False, expert_normal={},
            dense_linear=lambda x, w, b: router_output,
            expert_linear=lambda *args, **kwargs: torch.zeros(1, 2),
            expert_activation=lambda up, limit: up,
            combine=lambda down, probabilities: captured.update(probabilities=probabilities) or down)

    def test_overflowing_router_logits_fail_before_softmax(self):
        captured = {}
        fake = self._fake_executor(torch.tensor([[float('inf'), 5.0, 0.0, 0.0]]), captured)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite GPT-OSS routing logits'):
            GptOssExecutor.experts(fake, torch.zeros(1, 2), 0)
        self.assertNotIn('probabilities', captured)

    def test_finite_routing_reaches_softmax(self):
        captured = {}
        fake = self._fake_executor(torch.tensor([[2.0, 1.0, 0.0, 0.0]]), captured)
        GptOssExecutor.experts(fake, torch.zeros(1, 2), 0)
        expected = torch.tensor([[2.0, 1.0]]).softmax(-1)
        torch.testing.assert_close(captured['probabilities'], expected)


class VisionEagerAttentionTests(unittest.TestCase):
    def test_overflow_fails_and_ordinary_matches_reference(self):
        torch.manual_seed(11)
        query = torch.randn(1, 2, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        values = torch.randn(1, 2, 3, 8)
        expected = (query @ key.transpose(-1, -2)) * 8 ** -.5
        torch.testing.assert_close(_eager_attention(query, key, values, 8),
                                   expected.softmax(-1) @ values)
        hot_q = query.clone(); hot_q[0, 0, 0, 0] = 1e20
        hot_k = key.clone(); hot_k[0, 0, 0, 0] = 1e20
        with self.assertRaisesRegex(RuntimeError, 'nonfinite vision attention scores'):
            _eager_attention(hot_q, hot_k, values, 8)


if __name__ == '__main__':
    unittest.main()
