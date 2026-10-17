"""Run layout selection gates against CPU fake executors, never GPU hardware."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from scripts import verify_gguf_layout_runtime as verify
case=sys.argv[1]
class Executor:
    kernel_backend='fixture+cuda-q4-marlin'
    q4_marlin_abi=2;packed_layout_overhead_bytes=0
    compute_dtype=residual_dtype='torch.float32'
    context=2048 if case=='context' else 512
    prefill_token_budget=1024
    def __init__(self,model,**kwargs):
        self.quant_linear=NS(q4_marlin=[None]*129)
        stage='batch' if kwargs['max_batch']==4 else 'eager'
        kind=case.removesuffix('-'+stage)
        if kind!=case or case in ('context','budget'):
            if kind=='backend':self.kernel_backend='wrong'
            if kind=='count':self.quant_linear.q4_marlin=[]
            if kind=='abi':self.q4_marlin_abi=3
            if kind=='overhead':self.packed_layout_overhead_bytes=1
            if kind=='context':self.context=512
            if kind=='budget':self.prefill_token_budget=512
    def scores(self,n=1):return verify.torch.tensor([[0.,2.,1.]]*n)
    def prefill(self,ids):return self.scores()
    def decode(self,token):return self.scores(),None
    def prefill_batch(self,prompts,slots):return self.scores(len(prompts))
    def decode_batch(self,tokens,slots):return self.scores(len(tokens)),None
    def prefill_chunk(self,*args,**kwargs):return self.scores()
    def release_request(self,*args):pass
model=NS(raw_tokens=lambda text:[1,2],tokenizer=object(),bytes=16)
modules={'experiments.q4_marlin_context':NS(install_validation_context=Mock(return_value={})),
         'experiments.q4_marlin_prefill_budget':NS(install_validation_budget=Mock(return_value={}))}
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'model';path.write_bytes(b'fixture')
    output=Path(tmp)/'report.json'
    args=['verify',str(path),'--output',str(output),'--require-q4-marlin']
    if case=='context':args+=['--context','2048','--experimental-q4-marlin-context','2048']
    if case=='budget':args+=['--experimental-q4-marlin-prefill-budget','1024']
    try:
        with patch('sys.argv',args),patch.dict(os.environ,{'DAMENDAEMON_Q4_MARLIN_LIBRARY':str(path)},clear=True), \
                patch.dict(sys.modules,modules),patch.object(verify,'require_gpu'), \
                patch.object(verify,'GgufWeights',return_value=model),patch.object(verify,'make_executor',side_effect=Executor), \
                patch.object(verify,'device_name',return_value='fixture'),patch.object(verify,'empty_cache'):
            verify.main()
    finally:
        print('report_passed='+str(json.loads(output.read_text())['passed'] if output.exists() else None),file=sys.stderr)
'''


class GgufLayoutVerifierTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)

    def test_requested_layout_mismatch_fails_with_failed_evidence(self):
        for optimized in (False,True):
            for case in [kind+'-'+stage for kind in ('backend','count','abi','overhead') for stage in ('eager','batch')]+['context','budget']:
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertIn('report_passed=False',result.stderr)

    def test_valid_control_in_both_modes(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('report_passed=True',result.stderr)
