import unittest
from unittest.mock import patch

import torch

from dllm.cache_indices import prefill_indices, ring_decode_indices
from dllm.selected_attention import selected_attention


class CacheIndexTests(unittest.TestCase):
    def test_prefill_matches_causal_logical_history_at_boundaries(self):
        for length in (1, 3, 4, 5, 8, 13):
            for window in (1, 4, 16):
                for ratio in (0, 4, 8):
                    actual = prefill_indices(length, window=window, ratio=ratio, batch=2)
                    for position in range(length):
                        expected = list(range(max(0, position-window+1), position+1))
                        expected += [length+i for i in range((position+1)//ratio)] if ratio else []
                        for batch in range(2):
                            self.assertEqual([x for x in actual[batch, position].tolist() if x >= 0], expected)

    def test_chunked_query_plans_keep_full_sequence_compression_offset(self):
        full = prefill_indices(19, window=4, ratio=4)
        for start, stop in ((0, 3), (3, 8), (8, 19)):
            chunk = prefill_indices(19, window=4, ratio=4, query_start=start, query_stop=stop)
            for offset, position in enumerate(range(start, stop)):
                self.assertEqual([x for x in chunk[0, offset].tolist() if x >= 0],
                                 [x for x in full[0, position].tolist() if x >= 0])

    def test_ring_wrap_reordered_slots_and_compressed_visibility(self):
        positions = [17, 0, 3, 4, 7, 8, 2]
        actual = ring_decode_indices(positions, window=4, ratio=4)
        for row, position in enumerate(positions):
            expected = [x % 4 for x in range(max(0, position-3), position+1)]
            expected += [4+i for i in range((position+1)//4)]
            self.assertEqual([x for x in actual[row, 0].tolist() if x >= 0], expected)

    def test_ring_and_prefill_attention_agree_with_same_visible_vectors(self):
        rng = torch.Generator().manual_seed(13)
        raw = torch.randn(1, 13, 4, generator=rng)
        compressed = torch.randn(1, 3, 4, generator=rng)
        query = torch.randn(1, 13, 2, 4, generator=rng)
        sink = torch.tensor([.2, -.4])
        reference = selected_attention(query, torch.cat((raw, compressed), 1),
                                       prefill_indices(13, window=4, ratio=4), sink)
        ring = torch.full((1, 7, 4), float('nan'))
        for position in range(13):
            ring[:, position % 4] = raw[:, position]
            if (position+1) % 4 == 0:
                block = (position+1)//4-1
                ring[:, 4+block] = compressed[:, block]
            actual = selected_attention(query[:, position:position+1], ring,
                                        ring_decode_indices([position], window=4, ratio=4), sink)
            torch.testing.assert_close(actual, reference[:, position:position+1])

    def test_invalid_geometry_and_budget_rejected_before_allocation(self):
        with patch('dllm.cache_indices.torch.arange', side_effect=AssertionError('allocation started')):
            for options in ({'window':0}, {'window':True}, {'window':4, 'ratio':-1},
                            {'window':4, 'query_start':5, 'query_stop':4}, {'window':4, 'batch':0}):
                with self.assertRaises(ValueError):
                    prefill_indices(16, **options)
            with self.assertRaises(ValueError):
                prefill_indices(32768, window=128, ratio=4)
            for positions in ([], [[1, 2]], [True], [-1]):
                with self.assertRaises(ValueError):
                    ring_decode_indices(positions, window=4)
            with self.assertRaises(ValueError):
                ring_decode_indices([2**30], window=4, ratio=4)
