"""Chunked reference conversion must equal the independent whole-matrix result."""
import importlib.util
import unittest
import torch
from scripts.reference_model import roundtrip_reference_weight


@unittest.skipUnless(importlib.util.find_spec('gguf'), 'optional GGUF reference dependency required')
class ReferenceRoundtripTests(unittest.TestCase):
    def test_chunk_boundaries_preserve_upstream_quantization(self):
        from gguf.quants import quantize, dequantize
        from gguf.constants import GGMLQuantizationType
        generator = torch.Generator().manual_seed(912)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            weight = torch.randn(259, 96, generator=generator).to(dtype)
            weight[0].zero_()
            weight[127, :32] = 1
            weight[128, :32] = -1
            original = weight.clone()
            for mode in ('q8_0', 'q4_0'):
                kind = getattr(GGMLQuantizationType, mode.upper())
                expected = torch.from_numpy(dequantize(quantize(weight.float().numpy(), kind), kind)).to(dtype)
                actual = roundtrip_reference_weight(weight, mode)
                self.assertTrue(torch.equal(expected, actual), (dtype, mode))
                self.assertTrue(torch.equal(weight, original))
                self.assertNotEqual(weight.data_ptr(), actual.data_ptr())

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            roundtrip_reference_weight(torch.zeros(2, 32), 'q8_0', 0)
