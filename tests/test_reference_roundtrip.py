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

    def test_text_reference_does_not_quantize_vision(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.reference_model import load_reference
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.language_model = torch.nn.Linear(32, 8, bias=False)
        model.model.visual = torch.nn.Module()
        model.model.visual.projection = torch.nn.Linear(4304, 8, bias=False)
        original_visual = model.model.visual.projection.weight
        expected_text = roundtrip_reference_weight(model.model.language_model.weight, 'q8_0')
        loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
            with patch.dict('sys.modules', {'transformers': SimpleNamespace(Qwen3_5ForConditionalGeneration=loader)}):
                actual = load_reference(directory, 'q8_0')
        self.assertIs(actual.model.visual.projection.weight, original_visual)
        self.assertTrue(torch.equal(actual.model.language_model.weight, expected_text))

    def test_packed_fp32_reference_matches_whole_matrix(self):
        from gguf.quants import quantize, dequantize
        from gguf.constants import GGMLQuantizationType
        from scripts.reference_packed_cpu import PackedReferenceLinear
        torch.manual_seed(914)
        original = torch.nn.Linear(96, 259, bias=True).bfloat16()
        inputs = torch.randn(3, 96)
        for mode in ('q8_0', 'q4_0'):
            kind = getattr(GGMLQuantizationType, mode.upper())
            weights = torch.from_numpy(dequantize(quantize(original.weight.detach().float().numpy(), kind), kind))
            expected = torch.nn.functional.linear(inputs, weights, original.bias.float())
            packed = PackedReferenceLinear(original, mode).float()
            self.assertTrue(torch.equal(expected, packed(inputs)))
            self.assertLess(packed.packed.numel(), original.weight.numel()*2)
            with self.assertRaises(ValueError):
                packed(inputs.half())

    def test_packed_model_conversion_keeps_no_full_linear_weights(self):
        from scripts.reference_packed_cpu import replace_linears, PackedReferenceLinear
        model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32)).bfloat16()
        expected = torch.nn.Sequential(
            PackedReferenceLinear(model[0], 'q8_0'), torch.nn.SiLU(),
            PackedReferenceLinear(model[2], 'q8_0')).float()
        actual = replace_linears(model, 'q8_0')
        inputs = torch.randn(2, 32)
        self.assertTrue(torch.equal(expected(inputs), actual(inputs)))
        self.assertEqual(sum(p.numel() for p in actual.parameters()), 0)

    def test_packed_gpu_reference_rejects_cpu_arithmetic(self):
        from scripts.reference_packed_cpu import PackedGpuReferenceLinear, replace_linears
        model = torch.nn.Sequential(torch.nn.Linear(32, 64)).bfloat16()
        result = replace_linears(model, 'q8_0', gpu_arithmetic=True)
        self.assertIsInstance(result[0], PackedGpuReferenceLinear)
        with self.assertRaises(ValueError):
            result(torch.ones(2, 32))

    def test_reference_rejects_conflicting_storage_modes(self):
        from scripts.reference_model import load_reference
        with self.assertRaises(ValueError):
            load_reference('/unused', packed_cpu=True, packed_gpu=True)

    def test_torch_expansion_matches_upstream_packing_and_extreme_codes(self):
        from gguf.quants import quantize, dequantize
        from gguf.constants import GGMLQuantizationType
        from scripts.reference_packed_cpu import expand_packed_torch
        generator = torch.Generator().manual_seed(915)
        for kind, size in ((GGMLQuantizationType.Q8_0, 34),
                           (GGMLQuantizationType.Q4_0, 18)):
            weight = torch.randn(259, 96, generator=generator)
            packed = torch.from_numpy(quantize(weight.numpy(), kind))
            expected = torch.from_numpy(dequantize(packed.numpy(), kind))
            self.assertTrue(torch.equal(expand_packed_torch(packed, kind), expected))
            # Raw blocks exercise every signed byte/nibble and finite scale
            # boundaries, including negative and subnormal FP16 scales.
            scales = torch.tensor([0, -0., 2**-24, -2**-24, 1, -1, 65504, -65504], dtype=torch.float16)
            blocks = torch.empty(8, size, dtype=torch.uint8)
            blocks[:, :2] = scales.view(torch.uint8).reshape(8, 2)
            blocks[:, 2:] = torch.arange(8 * (size-2)).remainder(256).to(torch.uint8).reshape(8, size-2)
            expected = torch.from_numpy(dequantize(blocks.numpy(), kind))
            self.assertTrue(torch.equal(expand_packed_torch(blocks, kind), expected))

    def test_torch_expansion_rejects_malformed_storage(self):
        from gguf.constants import GGMLQuantizationType as Q
        from scripts.reference_packed_cpu import expand_packed_torch
        for packed, kind in ((torch.zeros(1, 34), Q.Q8_0),
                             (torch.zeros(34, dtype=torch.uint8), Q.Q8_0),
                             (torch.zeros(1, 33, dtype=torch.uint8), Q.Q8_0),
                             (torch.zeros(34, 2, dtype=torch.uint8).T, Q.Q8_0),
                             (torch.zeros(1, 34, dtype=torch.uint8), Q.F32)):
            with self.assertRaises(ValueError):
                expand_packed_torch(packed, kind)
