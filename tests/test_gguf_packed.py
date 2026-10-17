import importlib.util
import io
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from damendaemon.gguf_packed import FORMATS, read_raw, layout


@unittest.skipUnless(importlib.util.find_spec('gguf'), 'optional gguf codec is not installed')
class PackedLayoutTests(unittest.TestCase):
    def fixture(self,kind):
        block,size=FORMATS[kind]
        raw=np.zeros((2*256//block,size),dtype=np.uint8)
        start=208 if kind==14 else 0
        count=2 if kind in (12,13) else 1
        raw[:,start:start+count*2]=np.full((len(raw),count),.001,dtype=np.float16).view(np.uint8)
        info=SimpleNamespace(kind=kind,offset=3,length=raw.nbytes,shape=(2,256),name='token_embd.weight')
        return raw,info,start,count

    def test_layout_preserves_native_bytes_and_mixed_row_offsets(self):
        parts=[]
        for kind in FORMATS:
            raw,info,_,_=self.fixture(kind)
            weight=read_raw(io.BytesIO(b'pad'+raw.tobytes()),info,embedding=True)
            self.assertTrue(np.array_equal(weight.numpy().reshape(raw.shape),raw))
            parts.append((weight,kind))
        width,spec=layout(256,parts)
        self.assertEqual(width,256)
        self.assertEqual([(s[0],s[1]) for s in spec],[(i*2,i*2+2) for i in range(5)])
        self.assertEqual([s[3] for s in spec],list(np.cumsum([0]+[t.numel() for t,_ in parts[:-1]])))
        value,kind=parts[0]
        self.assertEqual(layout(256,[(value,kind),(value,kind)])[1],((0,4,kind,0),))
        with self.assertRaises(ValueError):layout(255,parts)
        with self.assertRaises(ValueError):layout(256,[])
        with self.assertRaises(ValueError):layout(256,[(torch.zeros(2,256),1)])

    def test_truncated_and_nonfinite_blocks_fail(self):
        for kind in FORMATS:
            raw,info,start,_=self.fixture(kind)
            with self.assertRaises(ValueError):read_raw(io.BytesIO(b'pad'+raw.tobytes()[:-1]),info)
            raw[0,start:start+2]=np.array([float('nan')],dtype=np.float16).view(np.uint8)
            with self.assertRaisesRegex(ValueError,'non-finite scales'):
                read_raw(io.BytesIO(b'pad'+raw.tobytes()),info)

    def test_extreme_embedding_scales_require_actual_finiteness(self):
        for kind in FORMATS:
            raw,info,start,count=self.fixture(kind)
            raw[:,start:start+count*2]=np.full((len(raw),count),65504,dtype=np.float16).view(np.uint8)
            if kind==2:raw[:,2:]=0x88
            read_raw(io.BytesIO(b'pad'+raw.tobytes()),info,embedding=True)
            if kind in (2,8):raw[0,2:]=0 if kind==2 else 127
            elif kind==14:raw[0,192]=127
            else:
                raw[0,4]=63
                raw[0,16 if kind==12 else 48]=255
            with self.assertRaisesRegex(ValueError,'finite FP16'):
                read_raw(io.BytesIO(b'pad'+raw.tobytes()),info,embedding=True)
