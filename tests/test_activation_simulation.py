import math
import unittest
from unittest.mock import patch

import torch

from damendaemon.activation_simulation import simulate_fp4, simulate_fp8, normalized_hadamard
from damendaemon.floating_compressor import compress_hidden
from damendaemon.deepseek_position import frequencies


class ActivationSimulationTests(unittest.TestCase):
    def test_fp4_midpoints_signed_zero_and_scalar_blocks(self):
        points = [0., -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6., -.25, -.75, -5., -6., .3, -.3]*2
        values = torch.tensor([points, [v*.03125 for v in points]])
        levels = [0., .5, 1., 1.5, 2., 3., 4., 6.]
        expected = []
        for row in values.tolist():
            scale = 2**math.ceil(math.log2(max(max(abs(v) for v in row), 6*2**-126)/6))
            codes = [min(range(8), key=lambda i:(abs(levels[i]-abs(v/scale)), i % 2)) for v in row]
            expected.append([math.copysign(levels[code]*scale, v) for v, code in zip(row, codes)])
        actual = simulate_fp4(values)
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
        torch.testing.assert_close(torch.signbit(actual), torch.signbit(torch.tensor(expected)))
        torch.testing.assert_close(simulate_fp4(torch.zeros(2, 32)), torch.zeros(2, 32))
        tiny = torch.tensor([levels*4])*2**-126
        torch.testing.assert_close(simulate_fp4(tiny), tiny, rtol=0, atol=0)

    def test_fp8_blocks_against_scalar_scale_and_dtype_rounding(self):
        values = torch.linspace(-1, 1, 256).reshape(2, 128)
        for block in (64, 128):
            expected = values.clone()
            for row in range(2):
                for start in range(0, 128, block):
                    part = values[row, start:start+block]
                    amax = max(max(abs(v) for v in part.tolist()), 1e-4)
                    scale = 2**math.ceil(math.log2(amax/448))
                    expected[row, start:start+block] = (part/scale).to(torch.float8_e4m3fn).float()*scale
            torch.testing.assert_close(simulate_fp8(values, block_size=block), expected, rtol=0, atol=0)
        torch.testing.assert_close(simulate_fp8(torch.zeros(2, 64)), torch.zeros(2, 64))

    def test_hadamard_matches_explicit_matrix_and_inverse(self):
        matrix = torch.ones(1, 1)
        for _ in range(5):
            matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
        values = torch.arange(64, dtype=torch.float32).reshape(2, 32)/32
        expected = values@matrix.T/math.sqrt(32)
        torch.testing.assert_close(normalized_hadamard(values), expected)
        torch.testing.assert_close(normalized_hadamard(normalized_hadamard(values)), values)

    def test_compressor_transform_modes_and_chunk_equivalence(self):
        rng = torch.Generator().manual_seed(101)
        for mode, width, rotary in (('kv_fp8', 128, 64), ('index_fp4', 32, 16)):
            hidden = torch.randn(1, 9, 5, generator=rng).bfloat16()
            parameters = [torch.randn(2*width, 5, generator=rng), torch.randn(2*width, 5, generator=rng),
                          torch.randn(4, 2*width, generator=rng), torch.randn(width, generator=rng), frequencies(rotary, base=160000)]
            plain, _, _ = compress_hidden(hidden, *parameters)
            actual, _, _ = compress_hidden(hidden, *parameters, cache_transform=mode)
            if mode == 'kv_fp8':
                expected = plain.clone()
                expected[..., :-rotary] = simulate_fp8(plain[..., :-rotary])
                torch.testing.assert_close(actual[..., -rotary:], plain[..., -rotary:], rtol=0, atol=0)
            else:
                expected = simulate_fp4(normalized_hadamard(plain))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            outputs, state = [], None
            for start in range(0, 9, 3):
                out, state, _ = compress_hidden(hidden[:, start:start+3], *parameters, state=state, cache_transform=mode)
                outputs.append(out)
            torch.testing.assert_close(torch.cat(outputs, 1), actual, rtol=0, atol=0)

    def test_validation_dtype_immutability_and_no_host_extraction(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            values = torch.linspace(-3, 3, 128).to(dtype)
            before = values.clone()
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                    patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')), \
                    patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
                for fn in (simulate_fp4, simulate_fp8, normalized_hadamard):
                    self.assertEqual(fn(values).dtype, dtype)
            torch.testing.assert_close(values, before)
        for fn in (simulate_fp4, simulate_fp8, normalized_hadamard):
            self.assertEqual(fn(torch.empty(2, 0, 64)).shape, (2, 0, 64))
            with self.assertRaises(ValueError):
                fn(torch.zeros(3))
            with self.assertRaises(ValueError):
                fn(torch.zeros(64, dtype=torch.int32))
