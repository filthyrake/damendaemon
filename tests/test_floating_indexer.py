import unittest
from unittest.mock import patch

import torch

from damendaemon.activation_simulation import normalized_hadamard, simulate_fp4
from damendaemon.compressed_indexer import select_compressed
from damendaemon.deepseek_position import frequencies, rotate_pairs
from damendaemon.floating_compressor import compress_hidden
from damendaemon.floating_indexer import index_hidden, checked_index_hidden


class FloatingIndexerTests(unittest.TestCase):
    def fixture(self, dtype=torch.bfloat16):
        rng = torch.Generator().manual_seed(107)
        hidden = torch.randn(2, 19, 5, generator=rng).to(dtype)
        low_rank = torch.randn(2, 19, 7, generator=rng).to(dtype)
        query_weight = torch.randn(3*32, 7, generator=rng).to(dtype)
        head_weight = torch.randn(3, 5, generator=rng).to(dtype)
        freq = frequencies(16, base=160000)
        parameters = [torch.randn(64, 5, generator=rng), torch.randn(64, 5, generator=rng),
                      torch.randn(4, 64, generator=rng), torch.randn(32, generator=rng), freq]
        keys, _, _ = compress_hidden(hidden, *parameters, cache_transform='index_fp4')
        positions = torch.arange(19).unsqueeze(0).expand(2, -1)
        return hidden, low_rank, query_weight, head_weight, keys, positions, freq

    def test_projection_transform_and_storage_scaled_head_coefficients(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden, low_rank, qw, hw, keys, positions, freq = args = self.fixture(dtype)
            queries = (low_rank@qw.T).reshape(2, 19, 3, 32)
            queries[..., -16:] = rotate_pairs(queries[..., -16:], positions, freq)
            queries = simulate_fp4(normalized_hadamard(queries))
            projected_weights = hidden@hw.T
            weights = (projected_weights*(32**-.5*3**-.5)).float()
            expected = select_compressed(queries, keys, weights, positions, ratio=4, top_k=3, offset=128, scale_head_weights=False)
            actual = index_hidden(*args, top_k=3, offset=128)
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=1e-5)
            if dtype != torch.float32:
                wrong_weights = projected_weights.float()*(32**-.5*3**-.5)
                self.assertFalse(torch.equal(weights, wrong_weights))

    def test_chunking_reordered_batches_and_empty_geometry(self):
        args = self.fixture()
        expected = index_hidden(*args, top_k=3)
        for chunk in (1, 3, 16):
            actual = index_hidden(*args, top_k=3, query_chunk=chunk, key_chunk=1)
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=1e-5)
        h, lr, qw, hw, keys, pos, freq = args
        reordered = index_hidden(h.flip(0), lr.flip(0), qw, hw, keys.flip(0), pos.flip(0), freq, top_k=3)
        torch.testing.assert_close(reordered[0], expected[0].flip(0))
        empty = index_hidden(h[:, :0], lr[:, :0], qw, hw, keys, pos[:, :0], freq, top_k=3)
        self.assertEqual(empty[0].shape, (2, 0, 3))
        no_keys = index_hidden(h, lr, qw, hw, keys[:, :0], pos, freq, top_k=3)
        self.assertEqual(no_keys[0].shape, (2, 19, 0))

    def test_no_host_reads_or_input_mutation_and_bounded_projection_rows(self):
        args = self.fixture()
        before = [x.clone() for x in args]
        original = torch.nn.functional.linear
        rows = []
        def record(x, weight):
            rows.append(x.shape[1])
            return original(x, weight)
        with patch('damendaemon.floating_indexer.linear', side_effect=record), \
                patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            index_hidden(*args, top_k=3, query_chunk=3)
        self.assertTrue(all(row <= 3 for row in rows))
        for value, previous in zip(args, before):
            torch.testing.assert_close(value, previous)

    def test_checked_reference_path_matches_valid_outputs_and_rejects_projection_overflow(self):
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            args=self.fixture(dtype)
            expected=index_hidden(*args,top_k=3)
            actual=checked_index_hidden(*args,top_k=3)
            torch.testing.assert_close(actual[0],expected[0])
            torch.testing.assert_close(actual[1],expected[1])
        args=list(self.fixture(torch.float32))
        args[1].fill_(2.)
        args[2].fill_(torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ValueError,'indexer intermediate'):
            checked_index_hidden(*args,top_k=3)

    def test_invalid_contracts_fail_before_projection(self):
        args = self.fixture()
        with patch('damendaemon.floating_indexer.linear', side_effect=AssertionError('projection started')):
            for options in ({'top_k':0}, {'top_k':3, 'offset':-1}, {'top_k':3, 'query_chunk':False}):
                with self.assertRaises(ValueError):
                    index_hidden(*args, **options)
            for index, replacement in ((2,args[2].float()), (3,args[3][:2]), (5,args[5].int()), (6,torch.ones(17))):
                changed = list(args); changed[index] = replacement
                with self.assertRaises(ValueError):
                    index_hidden(*changed, top_k=3)
