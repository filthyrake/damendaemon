from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from damendaemon.packed_checkpoint import read_packed_matrix
from damendaemon.prepared_projection import PackedProjection
from tests.test_packed_q4_q8 import fixture


class PackedIntegerCheckpointTests(unittest.TestCase):
    def test_typed_same_and_cross_shard_execution_and_owned_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.safetensors',Path(directory)/'b.safetensors'
            for format in ('q4_0_f16','q8_0_f16'):
                weight,scales,dense=fixture(format);hidden=torch.arange(192).reshape(2,96).float()/192
                for same in (False,True):
                    save_file(dict(w=weight,**({'s':scales} if same else {})),a)
                    if not same:save_file({'s':scales},b)
                    size=weight.numel()+2*scales.numel()
                    result=read_packed_matrix(a,'w',a if same else b,'s',format=format,max_bytes=size)
                    self.assertEqual(result.shape,(5,96));torch.testing.assert_close(result.weight,weight);torch.testing.assert_close(result.scales,scales)
                    actual=PackedProjection(result.weight,result.scales,format=format).linear(hidden)
                    torch.testing.assert_close(actual,hidden@dense.T,rtol=2e-5,atol=2e-5)
                    result.weight.zero_();result.scales.zero_()
                    reread=read_packed_matrix(a,'w',a if same else b,'s',format=format,max_bytes=size)
                    torch.testing.assert_close(reread.weight,weight);torch.testing.assert_close(reread.scales,scales)

    def test_storage_geometry_and_joint_budget_fail_before_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.safetensors'
            for format in ('q4_0_f16','q8_0_f16'):
                weight,scales,_=fixture(format)
                save_file({'w':weight,'s':scales},path)
                with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('payload')):
                    with self.assertRaisesRegex(ValueError,'budget'):
                        read_packed_matrix(path,'w',path,'s',format=format,max_bytes=weight.numel()+2*scales.numel()-1)
                    with patch('damendaemon.packed_checkpoint.sys.byteorder','big'):
                        with self.assertRaisesRegex(ValueError,'endian'):read_packed_matrix(path,'w',path,'s',format=format)
                    for w,s in ((weight,scales.float()),(weight.float(),scales),(weight,scales[:,:1].contiguous()),(weight[:,:-1].contiguous(),scales)):
                        save_file({'w':w,'s':s},path)
                        with self.assertRaises(ValueError):read_packed_matrix(path,'w',path,'s',format=format)

    def test_mutation_and_failed_read_close_both_descriptors(self):
        from damendaemon import packed_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.safetensors',Path(directory)/'b.safetensors'
            weight,scales,_=fixture('q8_0_f16');save_file({'w':weight},a);save_file({'s':scales},b)
            opened=[];header=packed_checkpoint._header;payload=packed_checkpoint._payload
            def record(reader):opened.append(reader);return header(reader)
            def mutate(*args):
                value=payload(*args)
                with b.open('ab') as writer:writer.write(b'x')
                return value
            for failure in (OSError('read failed'),mutate):
                save_file({'s':scales},b)
                with patch('damendaemon.packed_checkpoint._header',side_effect=record),patch('damendaemon.packed_checkpoint._payload',side_effect=failure):
                    with self.assertRaises((OSError,ValueError)):read_packed_matrix(a,'w',b,'s',format='q8_0_f16')
                self.assertTrue(all(reader.closed for reader in opened))
