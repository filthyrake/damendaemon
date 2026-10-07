import unittest
from unittest.mock import patch
import torch
from dllm.packed_tiles import decode_tile,validate_layout
from dllm.packed_projection import packed_linear
from dllm.prepared_projection import PackedProjection


def fixture(format,rows=5,columns=96):
    codes=torch.arange(rows*columns,dtype=torch.int64).reshape(rows,columns)
    if format=='q4_0_f16':
        codes=(codes*7+3)%16
        groups=codes.reshape(rows,-1,32)
        weight=(groups[:,:,:16]|(groups[:,:,16:]<<4)).to(torch.uint8).reshape(rows,columns//2)
    else:weight=((codes*37)%256-128).to(torch.int8)
    scales=(torch.arange(rows*(columns//32)).reshape(rows,columns//32)%7-3).to(torch.float16)*.125
    dense=torch.empty(rows,columns)
    for r in range(rows):
        for c in range(columns):
            if format=='q4_0_f16':
                code=int(weight[r,(c//32)*16+c%16]);integer=((code>>(4 if c%32>=16 else 0))&15)-8
            else:integer=int(weight[r,c])
            dense[r,c]=integer*float(scales[r,c//32])
    return weight,scales,dense


class PackedIntegerTests(unittest.TestCase):
    def test_scalar_decode_unaligned_blocks_negative_zero_and_strided_storage(self):
        for format in ('q4_0_f16','q8_0_f16'):
            weight,scales,dense=fixture(format)
            backing=torch.empty(weight.shape[0],weight.shape[1]*2,dtype=weight.dtype);backing[:,::2]=weight;weight=backing[:,::2]
            scale_backing=torch.empty(scales.shape[0],scales.shape[1]*2,dtype=torch.float16);scale_backing[:,::2]=scales;scales=scale_backing[:,::2]
            self.assertEqual(validate_layout(weight,scales,format=format),(5,96))
            for start,stop in ((0,96),(1,17),(15,33),(31,65),(63,96)):
                actual=decode_tile(weight,scales,format=format,row_start=1,row_stop=5,column_start=start,column_stop=stop)
                torch.testing.assert_close(actual,dense[1:5,start:stop],rtol=0,atol=0)

    def test_projection_and_row_views_never_expand_full_matrix(self):
        torch.manual_seed(1073);hidden=torch.randn(3,96)
        for format in ('q4_0_f16','q8_0_f16'):
            weight,scales,dense=fixture(format)
            original_weight=weight.clone();original_scales=scales.clone();shapes=[]
            def tile(*args,**kwargs):
                value=decode_tile(*args,**kwargs);shapes.append(value.shape);return value
            with patch('dllm.packed_projection.decode_tile',side_effect=tile):
                actual=packed_linear(hidden,weight,scales,format=format,row_chunk=2,column_chunk=17,workspace_bytes=16384)
                adapter=PackedProjection(weight,scales,format=format,row_chunk=2,column_chunk=17,workspace_bytes=16384).rows(1,4)
                viewed=adapter.linear(hidden.unsqueeze(0))
            torch.testing.assert_close(actual,hidden@dense.T,rtol=2e-5,atol=2e-5)
            torch.testing.assert_close(viewed,(hidden@dense[1:4].T).unsqueeze(0),rtol=2e-5,atol=2e-5)
            self.assertTrue(all(r<=2 and c<=17 for r,c in shapes))
            self.assertIs(adapter.weight,weight);self.assertIs(adapter.scales,scales)
            torch.testing.assert_close(weight,original_weight);torch.testing.assert_close(scales,original_scales)

    def test_special_scales_propagate_in_diagnostic_tiles(self):
        for format in ('q4_0_f16','q8_0_f16'):
            weight,scales,_=fixture(format);scales[0,0]=float('nan');scales[1,0]=float('inf')
            actual=decode_tile(weight,scales,format=format,row_start=0,row_stop=2,column_start=0,column_stop=32)
            self.assertTrue(torch.isnan(actual[0]).all());self.assertTrue(torch.isinf(actual[1]).any())

    def test_format_storage_shape_tile_and_workspace_rejection(self):
        for format in ('q4_0_f16','q8_0_f16'):
            weight,scales,_=fixture(format)
            for w,s in ((weight,scales.float()),(weight.float(),scales),(weight,scales[:,:1]),(weight[:,:-1],scales)):
                with self.assertRaises(ValueError):validate_layout(w,s,format=format)
            with patch('dllm.packed_projection.decode_tile',side_effect=AssertionError('decoded before admission')):
                with self.assertRaises(ValueError):packed_linear(torch.zeros(1,96),weight,scales,format=format,workspace_bytes=1)
            large=torch.zeros((257,256 if format=='q4_0_f16' else 512),dtype=weight.dtype)
            with self.assertRaisesRegex(ValueError,'65536'):decode_tile(large,torch.ones(257,16,dtype=torch.float16),format=format,row_start=0,row_stop=257,column_start=0,column_stop=256)
