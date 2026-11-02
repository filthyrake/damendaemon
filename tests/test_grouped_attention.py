import unittest
from unittest import mock

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
        # Compensated chunk prefill admits on every compensated-prefill
        # accelerator (ROCm HOMER bring-up; CUDA Q4 since #679 — the eager
        # SDPA path's score block grows linearly with the prefix there).
        self.assertTrue(use_grouped_prefill('rocm', 'compensated', True, 127, 4096))
        self.assertTrue(use_grouped_prefill('cuda', 'compensated', True, 127, 4096))
        for args in (('mps', 'compensated', True, 127, 4096),
                     ('rocm', 'float32', True, 127, 4096), ('cuda', 'float16', True, 127, 4096),
                     ('rocm', 'compensated', False, 127, 4096), ('cuda', 'compensated', False, 127, 4096),
                     ('rocm', 'compensated', True, 1, 4096), ('cuda', 'compensated', True, 1, 4096),
                     ('rocm', 'compensated', True, 127, 512), ('cuda', 'compensated', True, 512, 128)):
            self.assertFalse(use_grouped_prefill(*args))

    def test_ab_control_override_forces_the_sdpa_path(self):
        # DAMENDAEMON_HYBRID_GROUPED_PREFILL=0 is the qualification-cell A/B
        # control; anything but 0/1 is a configuration error.
        import os
        with mock.patch.dict(os.environ, {'DAMENDAEMON_HYBRID_GROUPED_PREFILL': '0'}):
            self.assertFalse(use_grouped_prefill('cuda', 'compensated', True, 127, 32768))
            self.assertFalse(use_grouped_prefill('rocm', 'compensated', True, 127, 32768))
        with mock.patch.dict(os.environ, {'DAMENDAEMON_HYBRID_GROUPED_PREFILL': '1'}):
            self.assertTrue(use_grouped_prefill('cuda', 'compensated', True, 127, 32768))
        with mock.patch.dict(os.environ, {'DAMENDAEMON_HYBRID_GROUPED_PREFILL': 'yes'}):
            with self.assertRaises(ValueError):
                use_grouped_prefill('cuda', 'compensated', True, 127, 32768)
