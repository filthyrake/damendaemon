import math
import unittest
from unittest.mock import patch

import torch

from damendaemon.selected_attention import selected_attention


def scalar_attention(query, kv, indices, sink, scale):
    result = torch.zeros_like(query, dtype=torch.float32)
    for b in range(query.shape[0]):
        for s in range(query.shape[1]):
            selected = [i for i in indices[b, s].tolist() if i >= 0]
            for h in range(query.shape[2]):
                scores = [sum(x*y for x, y in zip(query[b, s, h].tolist(), kv[b, i].tolist()))*scale
                          for i in selected]
                maximum = max([sink[h].item()]+scores)
                numerators = [math.exp(x-maximum) for x in scores]
                denominator = math.exp(sink[h].item()-maximum)+sum(numerators)
                for d in range(query.shape[3]):
                    result[b, s, h, d] = sum(p*kv[b, i, d].item() for p, i in zip(numerators, selected))/denominator
    return result.to(query.dtype)


class SelectedAttentionTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        rng = torch.Generator().manual_seed(41)
        query = torch.randn(2, 5, 3, 4, generator=rng).to(dtype)
        kv = torch.randn(2, 9, 4, generator=rng).to(dtype)
        indices = torch.randint(-1, 9, (2, 5, 7), generator=rng)
        indices[0, 0] = -1
        return query, kv, indices, torch.tensor([-.7, .3, 1.1])

    def test_scalar_reference_with_chunks_duplicates_and_storage_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            args = self.fixture(dtype)
            expected = scalar_attention(*args, scale=.3)
            for qchunk, kchunk in ((1, 1), (2, 3), (16, 64)):
                actual = selected_attention(*args, scale=.3, query_chunk=qchunk, key_chunk=kchunk)
                torch.testing.assert_close(actual, expected, rtol=2e-5 if dtype == torch.float32 else .01, atol=2e-6 if dtype == torch.float32 else .001)
                self.assertEqual(actual.dtype, dtype)

    def test_sink_is_denominator_only_and_large_scores_remain_finite(self):
        query = torch.tensor([[[[1000., 0.], [-1000., 0.]]]])
        kv = torch.tensor([[[1., 2.], [-1., 3.]]])
        indices = torch.tensor([[[0, 1, 0]]])
        sink = torch.tensor([1001., -1001.])
        actual = selected_attention(query, kv, indices, sink, scale=1., key_chunk=1)
        expected = scalar_attention(query, kv, indices, sink, 1.)
        torch.testing.assert_close(actual, expected)
        zero_query = torch.zeros(1, 1, 1, 2)
        actual = selected_attention(zero_query, kv[:, :1], torch.tensor([[[0]]]), torch.tensor([0.]))
        torch.testing.assert_close(actual, kv[:, :1, None, :]/2)

    def test_padding_empty_and_reordered_batches_without_host_extraction(self):
        args = self.fixture()
        query, kv, indices, sink = args
        originals = [x.clone() for x in args]
        expected = selected_attention(*args)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')):
            actual = selected_attention(query.flip(0), kv.flip(0), indices.flip(0), sink)
            empty = selected_attention(query[:, :0], kv, indices[:, :0], sink)
            zero_batch = selected_attention(query[:0], kv[:0], indices[:0], sink)
            no_keys = selected_attention(query, kv, indices[..., :0], sink)
            padding = selected_attention(query, torch.full_like(kv, float('nan')), torch.full_like(indices, -1), sink)
        torch.testing.assert_close(actual, expected.flip(0))
        self.assertEqual(empty.shape, (2, 0, 3, 4))
        self.assertEqual(zero_batch.shape, (0, 5, 3, 4))
        torch.testing.assert_close(no_keys, torch.zeros_like(query))
        torch.testing.assert_close(padding, torch.zeros_like(query))
        for tensor, original in zip(args, originals):
            torch.testing.assert_close(tensor, original)

    def test_gathers_stay_within_chunk_geometry(self):
        query, kv, indices, sink = self.fixture()
        shapes = []
        original = torch.matmul
        def record(a, b):
            shapes.append((a.shape, b.shape))
            return original(a, b)
        with patch('damendaemon.selected_attention.torch.matmul', side_effect=record):
            selected_attention(query, kv, indices, sink, query_chunk=2, key_chunk=3)
        self.assertEqual(len(shapes), 18)
        for index, (a, b) in enumerate(shapes):
            self.assertLessEqual(a[1], 2)
            self.assertLessEqual(b[-1] if index % 2 == 0 else b[-2], 3)

    def test_invalid_static_contracts_before_compute(self):
        query, kv, indices, sink = self.fixture()
        with patch('damendaemon.selected_attention.torch.matmul', side_effect=AssertionError('compute started')):
            for options in ({'query_chunk':0}, {'key_chunk':257}, {'key_chunk':True}, {'scale':float('inf')}, {'scale':0}):
                with self.assertRaises(ValueError):
                    selected_attention(query, kv, indices, sink, **options)
            for args in ((query, kv.half(), indices, sink), (query, kv, indices.float(), sink),
                         (query, kv, indices, sink[:1]), (query, kv[:, :0], indices, sink),
                         (query, kv, indices.to('meta'), sink)):
                with self.assertRaises(ValueError):
                    selected_attention(*args)
