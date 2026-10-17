import unittest
import torch
from torch.nn import functional as F
from damendaemon.grouped_attention import attention, use_grouped_prefill


class GroupedAttentionTests(unittest.TestCase):
    def test_causal_grouping_batch_strides_and_scratch_partitions(self):
        torch.manual_seed(68)
        for batch, heads, kv_heads, queries, context in ((2, 8, 2, 17, 769), (1, 6, 2, 31, 31), (1, 4, 4, 1, 33)):
            q = torch.randn(batch, queries, heads, 16).transpose(1, 2)
            k = torch.randn(batch, context, kv_heads, 16).transpose(1, 2)
            v = torch.randn_like(k)
            mask = torch.arange(context)[None, :] <= context-queries+torch.arange(queries)[:, None]
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            actual = attention(q, k, v, scratch_bytes=64*1024)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual, attention(q, k, v), atol=1e-6, rtol=1e-5)

    def test_budget_and_geometry_rejections(self):
        q = torch.randn(1, 4, 17, 16)
        k = torch.randn(1, 2, 33, 16)
        for budget in (0, -1, True, 1.5, 1):
            with self.assertRaises(ValueError): attention(q, k, k, scratch_bytes=budget)
        for query, key, value in ((q.half(), k, k), (q, k[:, :0], k[:, :0]),
                                  (q, k[:, :, :1], k[:, :, :1]), (q[:, :3], k, k),
                                  (q, k, k[:, :, :, :8])):
            with self.assertRaises(ValueError): attention(query, key, value)

    def test_dispatch_preserves_other_devices_precision_decode_and_short_context(self):
        self.assertTrue(use_grouped_prefill('rocm', 'compensated', True, 127, 4096))
        for args in (('cuda', 'compensated', True, 127, 4096), ('mps', 'compensated', True, 127, 4096),
                     ('rocm', 'float32', True, 127, 4096), ('rocm', 'compensated', False, 127, 4096),
                     ('rocm', 'compensated', True, 1, 4096), ('rocm', 'compensated', True, 127, 512)):
            self.assertFalse(use_grouped_prefill(*args))
