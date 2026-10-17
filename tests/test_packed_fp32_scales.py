from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.packed_tiles import decode_tile,validate_layout
from damendaemon.packed_projection import packed_linear
from damendaemon.prepared_projection import PackedProjection
from damendaemon.packed_checkpoint import read_packed_matrix
from tests.test_packed_checkpoint import write_raw


def scalar(code):
    exponent=(code>>3)&15;fraction=code&7
    value=(fraction*2**-9 if exponent==0 else (1+fraction/8)*2**(exponent-7))
    if exponent==15 and fraction==7:value=float('nan')
    return -value if code&128 else value


def fixture():
    raw=(torch.arange(130*132).reshape(130,132)%126).to(torch.uint8)
    scales=torch.tensor([[.3,.7],[1.25,.125]],dtype=torch.float32)
    dense=torch.tensor([[scalar(int(raw[r,c]))*float(scales[r//128,c//128]) for c in range(132)] for r in range(130)])
    return raw,scales,dense


def write_pair(path,raw,scales,*,both=True):
    weight=bytes(raw.flatten().tolist())
    scale=struct.pack('<'+'f'*scales.numel(),*scales.flatten().tolist())
    header={'w':dict(dtype='F8_E4M3',shape=list(raw.shape),data_offsets=[0,len(weight)])}
    if both:header['s']=dict(dtype='F32',shape=list(scales.shape),data_offsets=[len(weight),len(weight)+len(scale)])
    write_raw(path,header,weight+scale if both else weight)
    return scale


class PackedFp32ScaleTests(unittest.TestCase):
    def test_decode_cross_block_tiles_projection_and_row_views(self):
        raw,scales,dense=fixture()
        tile=decode_tile(raw,scales,format='e4m3fn_f32',row_start=125,row_stop=130,column_start=125,column_stop=132)
        torch.testing.assert_close(tile,dense[125:130,125:132],rtol=0,atol=0)
        hidden=torch.linspace(-.1,.2,264).reshape(2,132)
        result=packed_linear(hidden,raw,scales,format='e4m3fn_f32',row_chunk=31,column_chunk=47)
        torch.testing.assert_close(result,hidden@dense.T,rtol=2e-5,atol=1e-3)
        projection=PackedProjection(raw,scales,format='e4m3fn_f32',row_chunk=31,column_chunk=47).rows(125,130)
        torch.testing.assert_close(projection.linear(hidden),hidden@dense[125:130].T,rtol=2e-5,atol=1e-3)
        self.assertIs(projection.weight,raw);self.assertIs(projection.scales,scales)

    def test_scale_types_and_special_values_are_explicit(self):
        raw=torch.tensor([[0,56,184,127]],dtype=torch.uint8)
        for scale in (.25,0.,-2.,float('nan'),float('inf')):
            actual=decode_tile(raw,torch.tensor([[scale]]),format='e4m3fn_f32',row_start=0,row_stop=1,column_start=0,column_stop=4)
            expected=torch.tensor([[scalar(int(code))*scale for code in raw[0]]])
            torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        for format,scales in (('e4m3fn_f32',torch.ones(1,1,dtype=torch.uint8)),('e4m3fn',torch.ones(1,1)),('e4m3fn_f32',torch.ones(1,2))):
            with self.assertRaises(ValueError):validate_layout(raw,scales,format=format)

    def test_typed_same_cross_shard_reads_and_preflight_budget(self):
        raw,scales,dense=fixture()
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.safetensors',Path(directory)/'b.safetensors'
            for same in (True,False):
                payload=write_pair(a,raw,scales,both=same)
                if not same:write_raw(b,{'s':dict(dtype='F32',shape=[2,2],data_offsets=[0,16])},payload)
                needed=raw.numel()+scales.numel()*4
                with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('values read')):
                    with self.assertRaisesRegex(ValueError,'budget'):read_packed_matrix(a,'w',a if same else b,'s',format='e4m3fn_f32',max_bytes=needed-1)
                    with self.assertRaises(ValueError):read_packed_matrix(a,'w',a if same else b,'s',format='e4m3fn')
                pair=read_packed_matrix(a,'w',a if same else b,'s',format='e4m3fn_f32',max_bytes=needed)
                torch.testing.assert_close(pair.weight,raw,rtol=0,atol=0)
                torch.testing.assert_close(pair.scales,scales,rtol=0,atol=0)
                pair.scales.zero_()
                reread=read_packed_matrix(a,'w',a if same else b,'s',format='e4m3fn_f32')
                torch.testing.assert_close(reread.scales,scales,rtol=0,atol=0)
            with patch('damendaemon.packed_checkpoint.sys.byteorder','big'):
                with self.assertRaisesRegex(ValueError,'endian'):read_packed_matrix(a,'w',b,'s',format='e4m3fn_f32')
