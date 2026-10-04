import os
import unittest
from unittest.mock import patch
import torch
from dllm.quantization import quantize, dequantize, prepare, mode


class QuantizationTests(unittest.TestCase):
    def test_zero_blocks_and_rounding(self):
        x = torch.zeros(2,32)
        q,s = quantize(x)
        self.assertTrue(torch.equal(dequantize(q,s),x))
        x[0,:7] = torch.tensor([-127,-63.5,-.5,0,.5,63.5,127])
        q,s = quantize(x)
        self.assertEqual(q[0,:7].tolist(),[-127,-64,-1,0,1,64,127])
        self.assertEqual(s[0].item(),1)

    def test_quantizing_tied_head_preserves_embedding(self):
        embedding = torch.randn(64,96).half()
        weights = {'embed':embedding,'head':embedding,'norm':torch.ones(96)}
        with patch.dict(os.environ, {'DLLM_QUANTIZATION':'q8_0'}):
            scales = prepare(weights)
        self.assertIs(weights['embed'],embedding)
        self.assertEqual(weights['embed'].dtype,torch.float16)
        self.assertEqual(weights['head'].dtype,torch.int8)
        self.assertEqual(set(scales),{'head'})
        self.assertLess((dequantize(weights['head'],scales['head'])-embedding.float()).abs().max().item(),.04)

    def test_invalid_inputs_and_mode(self):
        for value in (torch.zeros(1,31),torch.zeros(32),torch.full((1,32),float('nan'))):
            with self.assertRaises(ValueError):
                quantize(value)
        with patch.dict(os.environ, {'DLLM_QUANTIZATION':'q4'}), self.assertRaises(ValueError):
            mode()
