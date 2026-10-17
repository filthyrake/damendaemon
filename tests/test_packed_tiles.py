import math
import unittest
from unittest.mock import patch

import torch

from damendaemon.packed_tiles import decode_tile


class PackedTileTests(unittest.TestCase):
    def test_all_e2m1_nibbles_and_e8m0_scales(self):
        weight = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]*2, dtype=torch.uint8).repeat(256, 1)
        scales = torch.arange(256, dtype=torch.uint8).reshape(256, 1)
        actual = decode_tile(weight, scales, format='e2m1', row_start=0, row_stop=256,
                             column_start=0, column_stop=32)
        magnitudes = (0, .5, 1, 1.5, 2, 3, 4, 6)
        expected = [[math.copysign(magnitudes[code % 8], -1 if code & 8 else 1)*math.ldexp(1., scale-127)
                     if scale < 255 else float('nan') for code in list(range(16))*2] for scale in range(256)]
        expected = torch.tensor(expected, dtype=torch.float64).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(actual[:255]), torch.signbit(expected[:255]))

    def test_all_e4m3_bytes_against_scalar_reference_and_native_decode(self):
        weight = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
        actual = decode_tile(weight, torch.full((1, 2), 127, dtype=torch.uint8), format='e4m3fn',
                             row_start=0, row_stop=1, column_start=0, column_stop=256)
        expected = []
        for code in range(256):
            exponent, fraction = (code >> 3) & 15, code & 7
            if exponent == 15 and fraction == 7:
                value = float('nan')
            elif exponent == 0:
                value = fraction*2**-9
            else:
                value = (1+fraction/8)*2**(exponent-7)
            expected.append(math.copysign(value, -1 if code & 128 else 1))
        expected = torch.tensor([expected])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(actual, weight.view(torch.float8_e4m3fn).float(), rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.signbit(actual[0, 128]))

    def test_unaligned_tiles_strided_storage_and_scale_boundaries(self):
        for format, columns in (('e2m1', 288), ('e4m3fn', 270)):
            width = columns//2 if format == 'e2m1' else columns
            weight = torch.arange(260*width*2).remainder(256).to(torch.uint8).reshape(260, width*2)[:, ::2]
            shape = (260, columns//32) if format == 'e2m1' else (3, 3)
            scales = torch.arange(math.prod(shape)*2).remainder(7).add(124).to(torch.uint8).reshape(shape[0], shape[1]*2)[:, ::2]
            before_weight, before_scales = weight.clone(), scales.clone()
            actual = decode_tile(weight, scales, format=format, row_start=126, row_stop=131, column_start=31, column_stop=260)
            expected = torch.empty_like(actual)
            for r in range(126, 131):
                for c in range(31, 260):
                    code = int(weight[r, c//2 if format == 'e2m1' else c])
                    if format == 'e2m1':
                        code = (code >> ((c % 2)*4)) & 15
                        value = math.copysign((0, .5, 1, 1.5, 2, 3, 4, 6)[code & 7], -1 if code & 8 else 1)
                        scale = int(scales[r, c//32])
                    else:
                        value = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
                        scale = int(scales[r//128, c//128])
                    expected[r-126, c-31] = value*2**(scale-127)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(weight, before_weight)
            torch.testing.assert_close(scales, before_scales)

    def test_budget_and_invalid_contracts_before_decode(self):
        weight = torch.zeros(257, 256, dtype=torch.uint8)
        scales = torch.full((3, 2), 127, dtype=torch.uint8)
        options = dict(format='e4m3fn', row_start=0, row_stop=256, column_start=0, column_stop=256)
        self.assertEqual(decode_tile(weight, scales, **options).numel(), 65536)
        invalid = ({'row_stop':257}, {'row_start':-1}, {'column_stop':257}, {'column_start':True},
                   {'row_stop':0}, {'format':'e5m2'})
        with patch('damendaemon.packed_tiles._e4m3', side_effect=AssertionError('decode started')):
            for override in invalid:
                with self.assertRaises(ValueError):
                    decode_tile(weight, scales, **(options | override))
            for w, s in ((weight.float(), scales), (weight, scales[:, :1]), (weight.to('meta'), scales)):
                with self.assertRaises(ValueError):
                    decode_tile(w, s, **options)
        with self.assertRaises(ValueError):
            decode_tile(torch.zeros(1, 17, dtype=torch.uint8), torch.zeros(1, 1, dtype=torch.uint8),
                        format='e2m1', row_start=0, row_stop=1, column_start=0, column_stop=1)
