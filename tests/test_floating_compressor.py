import math
import unittest
from unittest.mock import patch

import torch

from dllm.deepseek_position import frequencies, rotate_pairs
from dllm.floating_compressor import compress_hidden
from tests.test_compressed_pool import scalar_pool


class FloatingCompressorTests(unittest.TestCase):
    def fixture(self, ratio=4, dtype=torch.float32):
        rng = torch.Generator().manual_seed(97)
        channels = 8 if ratio == 4 else 4
        return (torch.randn(2, 19, 5, generator=rng).to(dtype),
                torch.randn(channels, 5, generator=rng), torch.randn(channels, 5, generator=rng),
                torch.randn(ratio, channels, generator=rng), torch.randn(4, generator=rng),
                frequencies(2, base=160000, original_length=65536, factor=16))

    def test_frequency_schedule_and_adjacent_rotation_scalar_reference(self):
        for original in (0, 65536):
            actual = frequencies(64, base=160000, original_length=original, factor=16)
            expected = []
            for i in range(32):
                value = 160000**(-2*i/64)
                if original:
                    low = max(math.floor(64*math.log(original/(32*2*math.pi))/(2*math.log(160000))), 0)
                    high = min(math.ceil(64*math.log(original/(2*math.pi))/(2*math.log(160000))), 63)
                    blend = max(0, min(1, (i-low)/(high-low if high != low else .001)))
                    value = value/16*blend+value*(1-blend)
                expected.append(value)
            torch.testing.assert_close(actual, torch.tensor(expected), rtol=2e-6, atol=1e-8)
        values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)/10
        positions = torch.tensor([[0, 2, 7], [4, 3, 1]])
        freq = frequencies(4, base=10000)
        result = rotate_pairs(values, positions, freq)
        expected = values.clone()
        for b in range(2):
            for s in range(3):
                for pair in range(2):
                    angle = positions[b, s].item()*freq[pair].item()
                    a, c = values[b, s, pair*2:pair*2+2].tolist()
                    expected[b, s, pair*2] = a*math.cos(angle)-c*math.sin(angle)
                    expected[b, s, pair*2+1] = c*math.cos(angle)+a*math.sin(angle)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(rotate_pairs(result, positions, freq, inverse=True), values)
        headed = rotate_pairs(values.unsqueeze(2).expand(-1, -1, 2, -1), positions, freq)
        torch.testing.assert_close(headed[:, :, 1], expected)

    def test_projection_pool_norm_rotation_reference_and_cast_order(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden, value, gate, offsets, norm, freq = args = self.fixture(dtype=dtype)
            output, _, starts = compress_hidden(*args)
            pooled = scalar_pool(hidden.float()@value.T, hidden.float()@gate.T, offsets)
            rounded = pooled.to(dtype).float()
            expected = (norm*rounded*(rounded.square().mean(-1, keepdim=True)+1e-6).rsqrt()).to(dtype)
            expected[..., -2:] = rotate_pairs(expected[..., -2:], starts, freq)
            torch.testing.assert_close(output, expected, rtol=2e-5 if dtype == torch.float32 else .01,
                                       atol=1e-6 if dtype == torch.float32 else .001)
            if dtype != torch.float32:
                late_cast = (norm*pooled*(pooled.square().mean(-1, keepdim=True)+1e-6).rsqrt()).to(dtype)
                late_cast[..., -2:] = rotate_pairs(late_cast[..., -2:], starts, freq)
                self.assertFalse(torch.equal(output, late_cast))

    def test_chunked_and_single_token_compression_match_full_prefill(self):
        for ratio in (4, 8):
            hidden, *parameters = self.fixture(ratio=ratio)
            expected, _, starts = compress_hidden(hidden, *parameters)
            for chunk in (1, 3, 7):
                state, outputs, positions = None, [], []
                for index in range(0, hidden.shape[1], chunk):
                    out, state, pos = compress_hidden(hidden[:, index:index+chunk], *parameters, state=state)
                    outputs.append(out); positions.append(pos)
                torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=2e-5, atol=1e-6)
                torch.testing.assert_close(torch.cat(positions), starts)

    def test_downstream_failure_preserves_prior_state(self):
        hidden, *parameters = self.fixture()
        _, state, _ = compress_hidden(hidden[:, :3], *parameters)
        before = state.values.clone(), state.scores.clone()
        with patch('dllm.floating_compressor.rotate_pairs', side_effect=RuntimeError('rotation failed')):
            with self.assertRaisesRegex(RuntimeError, 'rotation failed'):
                compress_hidden(hidden[:, 3:7], *parameters, state=state)
        torch.testing.assert_close(state.values, before[0])
        torch.testing.assert_close(state.scores, before[1])
        self.assertEqual(state.position, 3)

    def test_invalid_static_parameters_before_projection(self):
        hidden, value, gate, offsets, norm, freq = self.fixture()
        with patch('dllm.floating_compressor.F.linear', side_effect=AssertionError('projection started')):
            for args in ((hidden, value.half(), gate, offsets, norm, freq),
                         (hidden, value, gate, offsets, norm, torch.ones(3)),
                         (hidden, value, gate, offsets, norm[:1], freq)):
                with self.assertRaises(ValueError):
                    compress_hidden(*args)
        for options in ({'dim':3, 'base':10000}, {'dim':4, 'base':1}, {'dim':4, 'base':10000, 'factor':0}):
            with self.assertRaises(ValueError):
                frequencies(**options)
