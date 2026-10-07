import math
import unittest
import weakref
from unittest.mock import patch

import torch

from dllm.packed_projection import packed_linear
from dllm.packed_tiles import decode_tile


def fixture(format, rows=133, columns=160):
    rng = torch.Generator().manual_seed(151)
    width = columns//2 if format == 'e2m1' else columns
    raw = torch.randint(0, 256 if format == 'e2m1' else 119, (rows,width*2),generator=rng,dtype=torch.uint8)[:,::2]
    shape = (rows,columns//32) if format == 'e2m1' else ((rows+127)//128,(columns+127)//128)
    scales = torch.randint(124,129,shape,generator=rng,dtype=torch.uint8)
    dense = torch.empty(rows,columns)
    for row in range(rows):
        for column in range(columns):
            code = int(raw[row,column//2 if format == 'e2m1' else column])
            if format == 'e2m1':
                code = (code >> (4*(column%2))) & 15
                value = math.copysign((0,.5,1,1.5,2,3,4,6)[code&7],-1 if code&8 else 1)
                exponent = int(scales[row,column//32])-127
            else:
                exp, fraction = (code>>3)&15,code&7
                value = fraction*2**-9 if exp == 0 else (1+fraction/8)*2**(exp-7)
                value = math.copysign(value,-1 if code&128 else 1)
                exponent = int(scales[row//128,column//128])-127
            dense[row,column] = math.ldexp(value,exponent)
    hidden = torch.randn(7,columns*2,generator=rng)[:,::2]
    return hidden,raw,scales,dense


class PackedProjectionTests(unittest.TestCase):
    def test_scalar_weight_control_strides_scale_boundaries_and_dtypes(self):
        for format in ('e2m1','e4m3fn'):
            hidden,raw,scales,dense = fixture(format)
            for dtype in (torch.float32,torch.bfloat16,torch.float16):
                inputs = hidden.to(dtype)
                actual = packed_linear(inputs,raw,scales,format=format,row_chunk=17,column_chunk=31)
                expected = inputs.float()@dense.T
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-3)
                self.assertEqual(actual.dtype,torch.float32)

    def test_workspace_bounds_token_chunks_and_weight_tile_lifetime(self):
        hidden,raw,scales,dense = fixture('e2m1',rows=9,columns=64)
        calls = []
        previous = None
        def bounded(*args,**kwargs):
            nonlocal previous
            if previous is not None:
                self.assertIsNone(previous())
            calls.append((kwargs['row_stop']-kwargs['row_start'],kwargs['column_stop']-kwargs['column_start']))
            result = decode_tile(*args,**kwargs)
            previous = weakref.ref(result)
            return result
        budget = 128*(3*17+3+17)+8*17+8*3
        with patch('dllm.packed_projection.decode_tile',side_effect=bounded):
            actual = packed_linear(hidden,raw,scales,format='e2m1',row_chunk=3,column_chunk=17,workspace_bytes=budget)
        self.assertEqual(len(calls),12)
        self.assertTrue(all(r*c <= 51 for r,c in calls))
        torch.testing.assert_close(actual,hidden@dense.T,rtol=2e-5,atol=1e-5)
        with patch('dllm.packed_projection.decode_tile',side_effect=AssertionError('decode started')):
            with self.assertRaisesRegex(ValueError,'workspace'):
                packed_linear(hidden,raw,scales,format='e2m1',row_chunk=3,column_chunk=17,workspace_bytes=budget-1)

    def test_empty_and_invalid_inputs_do_not_decode(self):
        hidden,raw,scales,_ = fixture('e2m1',rows=3,columns=32)
        with patch('dllm.packed_projection.decode_tile',side_effect=AssertionError('decode started')):
            self.assertEqual(packed_linear(hidden[:0],raw,scales,format='e2m1').shape,(0,3))
            for override in ({'row_chunk':True},{'column_chunk':0},{'row_chunk':257,'column_chunk':256},
                             {'workspace_bytes':1},{'format':'unsupported'}):
                with self.assertRaises(ValueError):
                    packed_linear(hidden,raw,scales,**({'format':'e2m1'}|override))
            for bad in (hidden[:,:31],hidden.long(),hidden.to('meta'),hidden.unsqueeze(0)):
                with self.assertRaises(ValueError):
                    packed_linear(bad,raw,scales,format='e2m1')
            with self.assertRaises(ValueError):
                packed_linear(hidden,raw,scales[:,:0],format='e2m1')

    def test_special_values_propagate_and_inputs_are_unchanged(self):
        hidden = torch.ones(1,32)
        raw = torch.full((2,16),0x22,dtype=torch.uint8)
        scales = torch.tensor([[127],[255]],dtype=torch.uint8)
        original = raw.clone()
        actual = packed_linear(hidden,raw,scales,format='e2m1',column_chunk=7)
        self.assertEqual(actual[0,0],32)
        self.assertTrue(torch.isnan(actual[0,1]))
        torch.testing.assert_close(raw,original)
