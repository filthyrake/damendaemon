import math
import unittest
from unittest.mock import patch

import torch

from dllm.compressed_indexer import select_compressed


def oracle(query, keys, weights, positions, ratio, top, offset):
    batch, sequence, heads, width = query.shape
    count = min(top, keys.shape[1])
    ids = torch.full((batch, sequence, count), -1, dtype=torch.int64)
    scores = torch.full((batch, sequence, count), -float('inf'))
    for b in range(batch):
        for s in range(sequence):
            candidates = []
            for k in range(min(keys.shape[1], (positions[b, s].item()+1)//ratio)):
                score = sum(max(0, sum(x*y for x, y in zip(query[b, s, h].tolist(), keys[b, k].tolist())))
                            *weights[b, s, h].item()/math.sqrt(width*heads) for h in range(heads))
                candidates.append((score, k))
            candidates.sort(key=lambda entry: (-entry[0], entry[1]))
            for j, (score, key) in enumerate(candidates[:count]):
                ids[b, s, j], scores[b, s, j] = key+offset, score
    return ids, scores


class CompressedIndexerTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        rng = torch.Generator().manual_seed(83)
        query = torch.randn(2, 5, 3, 4, generator=rng).to(dtype)
        keys = torch.randn(2, 11, 4, generator=rng).to(dtype)
        weights = torch.randn(2, 5, 3, generator=rng)
        positions = torch.tensor([[0, 3, 7, 30, 43], [3, 2, 15, 44, 7]])
        return query, keys, weights, positions

    def test_scalar_scores_and_ids_across_chunks_and_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            args = self.fixture(dtype)
            expected = oracle(*args, 4, 5, 128)
            for qc, kc in ((1, 1), (2, 3), (16, 64)):
                actual = select_compressed(*args, ratio=4, top_k=5, offset=128, query_chunk=qc, key_chunk=kc)
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=1e-6)

    def test_ties_choose_lower_ids_and_future_poison_cannot_be_selected(self):
        query, keys, weights, positions = self.fixture()
        weights.zero_()
        for kc in (1, 3, 64):
            actual = select_compressed(query, keys, weights, positions, ratio=4, top_k=5, key_chunk=kc)
            expected = oracle(query, keys, weights, positions, 4, 5, 0)
            torch.testing.assert_close(actual[0], expected[0])
        # All rows can see only the first block. Uninitialized future keys may be NaN.
        positions.fill_(3)
        weights.fill_(-1)
        keys[:, 1:] = float('nan')
        ids, scores = select_compressed(query, keys, weights, positions, ratio=4, top_k=3, key_chunk=2)
        torch.testing.assert_close(ids[..., 0], torch.zeros(2, 5, dtype=torch.int64))
        self.assertTrue((ids[..., 1:] == -1).all())
        self.assertTrue(torch.isfinite(scores[..., 0]).all())
        self.assertTrue(torch.isneginf(scores[..., 1:]).all())

    def test_checked_dots_fail_before_relu_can_hide_overflow(self):
        query=torch.full((1,1,1,2),1e20)
        weights=torch.ones(1,1,1);positions=torch.zeros(1,1,dtype=torch.int64)
        for sign in (-1,1):
            keys=torch.full((1,1,2),sign*1e20)
            with self.subTest(sign=sign),self.assertRaisesRegex(ValueError,'dot product'):
                select_compressed(query,keys,weights,positions,ratio=1,top_k=1,validate_scores=True)
        with self.assertRaisesRegex(ValueError,'dot product'):
            select_compressed(torch.ones_like(query),torch.ones(1,1,2),weights,positions,
                ratio=1,top_k=1,dot_scale=torch.finfo(torch.float32).max,validate_scores=True)

    def test_checked_visibility_ignores_future_poison_and_matches_scalar_oracle(self):
        args=self.fixture()
        for qc,kc in ((1,1),(2,3),(16,64)):
            actual=select_compressed(*args,ratio=4,top_k=5,query_chunk=qc,key_chunk=kc,validate_scores=True)
            expected=oracle(*args,4,5,0)
            torch.testing.assert_close(actual[0],expected[0])
            torch.testing.assert_close(actual[1],expected[1],rtol=2e-5,atol=1e-6)
        query,keys,weights,positions=args
        positions.fill_(3);keys[:,1:]=float('nan')
        for kc in (1,3,64):
            ids,scores=select_compressed(query,keys,weights,positions,ratio=4,top_k=3,key_chunk=kc,validate_scores=True)
            self.assertTrue((ids[...,0]==0).all())
            self.assertTrue((ids[...,1:]==-1).all())
            self.assertTrue(torch.isfinite(scores[...,0]).all())
            self.assertTrue(torch.isneginf(scores[...,1:]).all())
        for flag in (None,1,'yes'):
            with self.assertRaises(ValueError):
                select_compressed(*args,ratio=4,top_k=1,validate_scores=flag)

    def test_reordered_empty_inputs_and_no_host_extraction(self):
        args = self.fixture()
        originals = [t.clone() for t in args]
        expected = select_compressed(*args, ratio=4, top_k=5)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            actual = select_compressed(*(t.flip(0) for t in args), ratio=4, top_k=5)
            query, keys, weights, positions = args
            empty = select_compressed(query, keys[:, :0], weights, positions, ratio=4, top_k=5)
            no_queries = select_compressed(query[:, :0], keys, weights[:, :0], positions[:, :0], ratio=4, top_k=5)
            no_batch = select_compressed(query[:0], keys[:0], weights[:0], positions[:0], ratio=4, top_k=5)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e.flip(0))
        self.assertEqual(empty[0].shape, (2, 5, 0))
        self.assertEqual(no_queries[0].shape, (2, 0, 5))
        self.assertEqual(no_batch[0].shape, (0, 5, 5))
        for tensor, original in zip(args, originals):
            torch.testing.assert_close(tensor, original)

    def test_score_geometry_stays_bounded_and_invalid_inputs_fail_early(self):
        args = self.fixture()
        original = torch.matmul
        shapes = []
        def record(a, b):
            shapes.append((a.shape, b.shape))
            return original(a, b)
        with patch('dllm.compressed_indexer.torch.matmul', side_effect=record):
            select_compressed(*args, ratio=4, top_k=5, query_chunk=2, key_chunk=3)
        self.assertEqual(len(shapes), 12)
        self.assertTrue(all(a[1] <= 2 and b[-1] <= 3 for a, b in shapes))
        with patch('dllm.compressed_indexer.torch.matmul', side_effect=AssertionError('compute started')):
            for options in ({'ratio':0, 'top_k':5}, {'ratio':4, 'top_k':True},
                            {'ratio':4, 'top_k':5, 'offset':-1}, {'ratio':4, 'top_k':5, 'key_chunk':257}):
                with self.assertRaises(ValueError):
                    select_compressed(*args, **options)
            query, keys, weights, positions = args
            with self.assertRaises(ValueError):
                select_compressed(query, keys, weights.half(), positions, ratio=4, top_k=5)
