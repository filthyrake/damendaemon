import unittest

import torch
from dllm.awq import Matrix, fuse, group_size, unpack_channels


class AwqFormatTests(unittest.TestCase):
    def matrix(self):
        weight = torch.tensor([0x76543210]*128 + [0xFEDCBA98 - 2**32]*128,
                              dtype=torch.int32).reshape(256, 1)
        zeros = torch.tensor([[0x11111111], [0x01234567]], dtype=torch.int32)
        scales = torch.tensor([[.5]*8, [.25]*8], dtype=torch.float16)
        return Matrix(weight, zeros, scales, 256, 8)

    def test_known_words_preserve_awq_channel_order_and_signed_high_nibble(self):
        words = torch.tensor([[0x76543210, 0xFEDCBA98 - 2**32]], dtype=torch.int32)
        self.assertEqual(unpack_channels(words).tolist(),
                         [[0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]])

    def test_group_boundary_zero_points_and_output_major_coefficients(self):
        matrix = self.matrix()
        before = matrix.weight.clone()
        expected = torch.tensor([[-.5, .25], [1.5, 2.25], [0., .75], [2., 2.75],
                                 [.5, 1.25], [2.5, 3.25], [1., 1.75], [3., 3.75]], dtype=torch.float16)
        decoded = matrix.dequantize()
        self.assertTrue(torch.equal(decoded, expected.repeat_interleave(128, dim=1)))
        self.assertTrue(torch.equal(matrix.weight, before))
        self.assertEqual(matrix.bytes, 1064)

    def test_fused_output_channels_keep_packed_storage_and_group_boundaries(self):
        left = self.matrix()
        right = Matrix(torch.zeros_like(left.weight), torch.zeros_like(left.zeros),
                       torch.ones_like(left.scales), 256, 8)
        merged = fuse([left, right])
        self.assertEqual(merged.weight.shape, (256, 2))
        self.assertEqual(merged.bytes, left.bytes + right.bytes)
        self.assertTrue(torch.equal(merged.dequantize(), torch.cat([left.dequantize(), right.dequantize()])))
        self.assertTrue(torch.equal(merged.weight[:, 0], left.weight[:, 0]))
        with self.assertRaises(ValueError):
            fuse([])

    def test_malformed_and_nonfinite_buffers_fail_before_use(self):
        m = self.matrix()
        for kwargs in ({'inputs': 255}, {'outputs': 7}, {'group': 128.0},
                       {'weight': m.weight.long()}, {'zeros': m.zeros[:1]},
                       {'scales': m.scales.float()}, {'scales': -m.scales},
                       {'scales': torch.full_like(m.scales, float('nan'))}):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                Matrix(**{**m.__dict__, **kwargs})

    def test_checkpoint_configuration_rejects_other_packing_and_exclusions(self):
        config = dict(bits=4, group_size=128, quant_method='awq', version='gemm',
                      zero_point=True, modules_to_not_convert=None)
        self.assertEqual(group_size(config), 128)
        self.assertEqual(group_size({**config, 'version': 'GEMM', 'modules_to_not_convert': []}), 128)
        for update in ({'bits': 4.0}, {'bits': 8}, {'group_size': 64}, {'quant_method': 'gptq'},
                       {'version': 'gemv'}, {'zero_point': False}, {'zero_point': 1},
                       {'modules_to_not_convert': ['lm_head']}, {'desc_act': True}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                group_size({**config, **update})
