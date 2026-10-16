"""Run CPU GGUF comparator mains with tiny tensors and real F32 unpacking."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from gguf import GGMLQuantizationType
from scripts import verify_decoder_gguf as dense, verify_mamba_gguf as mamba
family,case=sys.argv[1:]
weights={'embed':torch.arange(4,dtype=torch.float32).reshape(2,2),
         'head':torch.ones((2,2)),'norm':torch.ones(2)}
if case=='tied':weights['head']=weights['embed']
model=NS(config={'model_type':'qwen2' if family=='dense' else 'mamba','num_hidden_layers':0},
         weights=weights,scales={})
names={'embed':'token_embd.weight','head':'output.weight','norm':'output_norm.weight'}
tensors=[NS(name=names[key],data=value.numpy().copy(),tensor_type=GGMLQuantizationType.F32) for key,value in weights.items()]
if case=='tied':tensors=[t for t in tensors if t.name!='output.weight']
if case=='missing':tensors=[]
if case=='extra':tensors.append(NS(name='extra.weight',data=weights['norm'].numpy(),tensor_type=GGMLQuantizationType.F32))
if case=='duplicate':tensors.insert(0,NS(name=tensors[0].name,data=tensors[0].data+10,tensor_type=GGMLQuantizationType.F32))
if case=='value':tensors[0].data+=1
if case=='shape':tensors[0].data=tensors[0].data.reshape(1,4)
reader=NS(tensors=tensors,fields={'general.architecture':NS(contents=lambda:'wrong' if case=='architecture' else 'mamba')})
verify=dense if family=='dense' else mamba
loader='ModelWeights' if family=='dense' else 'load'
reader_owner=dense if family=='dense' else mamba.gguf
with patch('sys.argv',['verify','unused','fixture.gguf']),patch.object(verify,loader,return_value=model), \
        patch.object(reader_owner,'GGUFReader',return_value=reader):
    verify.main()
'''


class GgufComparatorGateTests(unittest.TestCase):
    def run_case(self,family,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,family,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)

    def test_invalid_evidence_does_not_publish_success(self):
        for optimized in (False,True):
            for family,cases in [('dense',('missing','duplicate','value')),('mamba',('architecture','extra','duplicate','value','shape'))]:
                for case in cases:
                    with self.subTest(family=family,case=case,optimized=optimized):
                        result=self.run_case(family,case,optimized)
                        self.assertNotEqual(result.returncode,0,result.stdout)
                        self.assertNotIn('"passed": true',result.stdout)

    def test_valid_and_tied_embedding_controls(self):
        for optimized in (False,True):
            for family,case in [('dense','valid'),('mamba','valid'),('mamba','tied')]:
                result=self.run_case(family,case,optimized)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('"passed": true',result.stdout)
