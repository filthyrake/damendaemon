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
