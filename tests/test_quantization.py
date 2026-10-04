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

    def test_q4_blocks_signed_scales_and_nibble_order(self):
        from dllm.quantization import quantize_q4,dequantize_q4
        weight=torch.arange(-8,8,dtype=torch.float32).repeat(2).reshape(1,32)
        data,scales=quantize_q4(weight)
        self.assertEqual(data.tolist(),[[i|(i<<4) for i in range(16)]])
        self.assertTrue(torch.equal(dequantize_q4(data,scales),weight))
        positive=-weight
        data,scales=quantize_q4(positive)
        self.assertEqual(scales.item(),-1)
        self.assertTrue(torch.equal(dequantize_q4(data,scales),positive))
        data,scales=quantize_q4(torch.zeros(2,64))
        self.assertTrue(torch.equal(dequantize_q4(data,scales),torch.zeros(2,64)))

    def test_q4_tied_head_and_invalid_weights(self):
        from dllm.quantization import quantize_q4
        embedding=torch.randn(64,96).half()
        weights={'embed':embedding,'head':embedding,'norm':torch.ones(96)}
        scales=prepare(weights,'q4_0')
        self.assertIs(weights['embed'],embedding)
        self.assertEqual(weights['head'].dtype,torch.uint8)
        self.assertEqual(weights['head'].shape,(64,48))
        self.assertEqual(scales['head'].shape,(64,3))
        for value in (torch.zeros(1,31),torch.zeros(32),torch.full((1,32),float('nan')),
                      torch.full((1,32),1.e20)):
            with self.assertRaises(ValueError):quantize_q4(value)
