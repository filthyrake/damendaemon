import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from damendaemon.packed_checkpoint import read_packed_matrix, MAX_HEADER_BYTES
from damendaemon.packed_projection import packed_linear
from tests.test_packed_projection import fixture


def write_shard(path,tensors):
    header,payload={},bytearray()
    for name,dtype,tensor in tensors:
        raw=bytes(tensor.contiguous().flatten().tolist())
        start=len(payload);payload.extend(raw)
        header[name]={'dtype':dtype,'shape':list(tensor.shape),'data_offsets':[start,len(payload)]}
    write_raw(path,header,payload)


def write_raw(path,header,payload):
    raw=header if isinstance(header,bytes) else json.dumps(header).encode()
    raw+=b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)


class PackedCheckpointTests(unittest.TestCase):
    def test_same_and_cross_shard_pairs_preserve_bytes_and_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.safetensors',Path(temp)/'b.safetensors'
            for format in ('e2m1','e4m3fn'):
                hidden,raw,scales,dense=fixture(format,9,64)
                dtype='I8' if format == 'e2m1' else 'F8_E4M3'
                for same in (True,False):
                    write_shard(a,[('w',dtype,raw)]+([('s','F8_E8M0',scales)] if same else []))
                    if not same:write_shard(b,[('s','F8_E8M0FNU',scales)])
                    result=read_packed_matrix(a,'w',a if same else b,'s',format=format,max_bytes=raw.numel()+scales.numel())
                    torch.testing.assert_close(result.weight,raw)
                    torch.testing.assert_close(result.scales,scales)
                    # Loaded buffers remain usable after both descriptors close.
                    actual=packed_linear(hidden,result.weight,result.scales,format=format)
                    torch.testing.assert_close(actual,hidden@dense.T,rtol=2e-5,atol=1e-3)
                    result.weight.zero_()
                    reread=read_packed_matrix(a,'w',a if same else b,'s',format=format)
                    torch.testing.assert_close(reread.weight,raw)

    def test_joint_budget_and_storage_preflight_before_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'a.safetensors'
            _,raw,scales,_=fixture('e2m1',3,32)
            write_shard(path,[('w','U8',raw),('s','F8_E8M0',scales)])
            with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('payload read')):
                with self.assertRaisesRegex(ValueError,'budget'):
                    read_packed_matrix(path,'w',path,'s',format='e2m1',max_bytes=raw.numel()+scales.numel()-1)
                for names,format in ((('missing','s'),'e2m1'),(('w','s'),'e4m3fn'),(('w','w'),'e2m1')):
                    with self.assertRaises(ValueError):
                        read_packed_matrix(path,names[0],path,names[1],format=format)
                write_shard(path,[('w','U8',raw),('s','F8_E8M0',torch.zeros(3,2,dtype=torch.uint8))])
                with self.assertRaisesRegex(ValueError,'geometry'):
                    read_packed_matrix(path,'w',path,'s',format='e2m1')

    def test_malformed_headers_ranges_and_truncation_rejected(self):
        valid={'w':{'dtype':'U8','shape':[1,16],'data_offsets':[0,16]},
               's':{'dtype':'F8_E8M0','shape':[1,1],'data_offsets':[16,17]}}
        cases=[]
        for change in ({'data_offsets':[0,1]},{'data_offsets':[17,18]}, {'shape':[True,1]},
                       {'dtype':'unsupported'}, {'shape':[1,2]}, {'extra':1}):
            altered=json.loads(json.dumps(valid));altered['s'].update(change);cases.append(altered)
        cases.extend([b'{"w":{},"w":{}}',b'[]',b'not-json',{'__metadata__':{'bad':1}}])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.safetensors'
            with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('payload read')):
                for header in cases:
                    write_raw(path,header,bytes(17))
                    with self.assertRaises(ValueError):read_packed_matrix(path,'w',path,'s',format='e2m1')
                write_raw(path,valid,bytes(16))
                with self.assertRaises(ValueError):read_packed_matrix(path,'w',path,'s',format='e2m1')
                for raw in (b'bad',struct.pack('<Q',MAX_HEADER_BYTES+1),struct.pack('<Q',128)+b'{}'):
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):read_packed_matrix(path,'w',path,'s',format='e2m1')

    def test_failure_closes_readers_and_midread_change_is_rejected(self):
        from damendaemon import packed_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.safetensors',Path(temp)/'b.safetensors'
            write_shard(a,[('w','U8',torch.zeros(1,16,dtype=torch.uint8))])
            write_shard(b,[('s','F8_E8M0',torch.full((1,1),127,dtype=torch.uint8))])
            readers=[]
            original_header=packed_checkpoint._header
            def record(reader):
                readers.append(reader)
                return original_header(reader)
            with patch('damendaemon.packed_checkpoint._header',side_effect=record), \
                    patch('damendaemon.packed_checkpoint._payload',side_effect=OSError('read failed')):
                with self.assertRaisesRegex(OSError,'read failed'):
                    read_packed_matrix(a,'w',b,'s',format='e2m1')
            self.assertEqual(len(readers),2)
            self.assertTrue(all(r.closed for r in readers))
            original_payload=packed_checkpoint._payload
            def modify(reader,base,entry):
                result=original_payload(reader,base,entry)
                with a.open('ab') as changed:changed.write(b'x')
                return result
            with patch('damendaemon.packed_checkpoint._payload',side_effect=modify):
                with self.assertRaisesRegex(ValueError,'changed'):
                    read_packed_matrix(a,'w',b,'s',format='e2m1')
