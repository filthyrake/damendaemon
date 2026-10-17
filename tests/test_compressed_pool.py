import math
import unittest
from unittest.mock import patch

import torch

from damendaemon.compressed_pool import PoolState, pool_projected


def scalar_pool(values, gates, offsets):
    batch, tokens, channels = values.shape
    ratio = offsets.shape[0]
    width = channels//2 if ratio == 4 else channels
    result = torch.empty(batch, tokens//ratio, width)
    for b in range(batch):
        for block in range(tokens//ratio):
            for d in range(width):
                pairs = []
                if ratio == 4 and block:
                    pairs.extend((values[b, t, d].item(), gates[b, t, d].item()+offsets[t % ratio, d].item())
                                 for t in range((block-1)*ratio, block*ratio))
                channel = d+width if ratio == 4 else d
                pairs.extend((values[b, t, channel].item(), gates[b, t, channel].item()+offsets[t % ratio, channel].item())
                             for t in range(block*ratio, (block+1)*ratio))
                maximum = max(score for _, score in pairs)
                weights = [math.exp(score-maximum) for _, score in pairs]
                result[b, block, d] = sum(value*w for (value, _), w in zip(pairs, weights))/sum(weights)
    return result


class CompressedPoolTests(unittest.TestCase):
    def fixture(self, ratio, tokens=19):
        rng = torch.Generator().manual_seed(61)
        channels = 6 if ratio == 4 else 3
        return (torch.randn(2, tokens, channels, generator=rng),
                torch.randn(2, tokens, channels, generator=rng), torch.randn(ratio, channels, generator=rng))

    def test_scalar_pooling_reference_with_and_without_overlap(self):
        for ratio in (1, 2, 4, 8, 128):
            values, gates, offsets = self.fixture(ratio, 259 if ratio == 128 else 19)
            actual, state, starts = pool_projected(values, gates, offsets)
            torch.testing.assert_close(actual, scalar_pool(values, gates, offsets), rtol=2e-5, atol=1e-6)
            self.assertEqual(state.position, values.shape[1])
            torch.testing.assert_close(starts, torch.arange(values.shape[1]//ratio)*ratio)

    def test_all_chunk_boundaries_and_state_are_equivalent(self):
        for ratio in (2, 4, 8):
            values, gates, offsets = self.fixture(ratio)
            expected, expected_state, expected_starts = pool_projected(values, gates, offsets)
            for step in (1, 3, 4, 7, 11):
                state = None
                results, positions = [], []
                for start in range(0, values.shape[1], step):
                    result, state, starts = pool_projected(values[:, start:start+step], gates[:, start:start+step], offsets, state=state)
                    results.append(result)
                    positions.append(starts)
                torch.testing.assert_close(torch.cat(results, 1), expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.cat(positions), expected_starts)
                torch.testing.assert_close(state.values, expected_state.values, rtol=0, atol=0)
                torch.testing.assert_close(state.scores, expected_state.scores, rtol=0, atol=0)

    def test_reordered_state_and_empty_chunk_preserve_inputs(self):
        values, gates, offsets = self.fixture(4)
        _, state, _ = pool_projected(values[:, :7], gates[:, :7], offsets)
        before = (state.values.clone(), state.scores.clone())
        expected, _, _ = pool_projected(values[:, 7:], gates[:, 7:], offsets, state=state)
        reordered = state.reorder(torch.tensor([1, 0, 1]))
        order = torch.tensor([1, 0, 1])
        actual, _, _ = pool_projected(values[order, 7:], gates[order, 7:], offsets, state=reordered)
        torch.testing.assert_close(actual, expected[order])
        empty, replacement, starts = pool_projected(values[:, :0], gates[:, :0], offsets, state=state)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(starts.numel(), 0)
        self.assertEqual(replacement.position, 7)
        for tensor, original in zip((state.values, state.scores), before):
            torch.testing.assert_close(tensor, original)
        self.assertNotEqual(replacement.values.data_ptr(), state.values.data_ptr())

    def test_late_failure_does_not_mutate_published_state(self):
        values, gates, offsets = self.fixture(4)
        _, state, _ = pool_projected(values[:, :3], gates[:, :3], offsets)
        before = (state.values.clone(), state.scores.clone())
        original = torch.Tensor.softmax
        calls = 0
        def fail_second(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('injected pooling failure')
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'softmax', fail_second):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                pool_projected(values[:, 3:], gates[:, 3:], offsets, state=state)
        for tensor, previous in zip((state.values, state.scores), before):
            torch.testing.assert_close(tensor, previous)
        self.assertEqual(state.position, 3)

    def test_shape_and_state_contracts_and_no_host_reads(self):
        values, gates, offsets = self.fixture(4)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')):
            pool_projected(values, gates, offsets)
        for v, g, o in ((values.half(), gates, offsets), (values, gates[:, :1], offsets),
                        (values, gates, offsets[:, :3]), (values[..., :5], gates[..., :5], offsets[:, :5])):
            with self.assertRaises(ValueError):
                pool_projected(v, g, o)
        bad = PoolState(torch.zeros(2, 8, 6), torch.zeros(2, 8, 6), -1, 4)
        with self.assertRaises(ValueError):
            pool_projected(values, gates, offsets, state=bad)
