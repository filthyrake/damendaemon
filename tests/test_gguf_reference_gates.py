"""Reference verifier acceptance with fake models and enforced CPU tensor creation."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
from contextlib import nullcontext
import json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from scripts import verify_gguf_reference as verify
case=sys.argv[1]
real_tensor=verify.torch.tensor
def cpu_tensor(*args,**kwargs):
    kwargs['device']='cpu'
    return real_tensor(*args,**kwargs)
class Reference:
    def load_state_dict(self,*args,**kwargs):pass
    def eval(self):return self
    def to(self,**kwargs):return self
    def __call__(self,*args,**kwargs):
        return NS(logits=real_tensor([[[0.,3.,1.,0.]]],device='cpu'),past_key_values=None)
class Executor:
    context=512;kernel_backend='fixture+cuda-q4-marlin'
    q4_marlin_abi=2;packed_layout_overhead_bytes=0
    compute_dtype=residual_dtype='torch.float32';precise=rotary_precise=False
    def __init__(self,*args,**kwargs):
        self.quant_linear=NS(q4_marlin=[None]*129)
        if case=='backend':self.kernel_backend='wrong'
        if case=='count':self.quant_linear.q4_marlin=[]
        if case=='abi':self.q4_marlin_abi=3
        if case=='overhead':self.packed_layout_overhead_bytes=1
    def prefill(self,*args):return real_tensor([[0.,0.,3.,1.]] if case=='numerical' else [[0.,3.,1.,0.]],device='cpu')
    def decode(self,*args):return self.prefill(),None
source=NS(config={'vocab_size':4},chat_tokens=lambda *args:[1],raw_tokens=lambda *args:[1],
          tokenizer=object(),bytes=16,eos=set())
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);model=root/'model';model.write_bytes(b'fixture')
    benchmark=root/'benchmark.json';benchmark.write_text(json.dumps(dict(prompts={'1':[1]},output_tokens=2)))
    output=root/'report.json'
    args=['verify',str(model),'unused-config','--output',str(output),'--benchmark-report',str(benchmark),'--require-q4-marlin']
    if case=='context':args+=['--context','2048','--experimental-q4-marlin-context','2048']
    try:
        with patch('sys.argv',args),patch.dict(os.environ,{'DLLM_Q4_MARLIN_LIBRARY':str(model)},clear=True), \
                patch.dict(sys.modules,{'experiments.q4_marlin_context':NS(install_validation_context=Mock(return_value={}))}), \
                patch.object(verify,'GgufWeights',return_value=source),patch.object(verify.AutoConfig,'from_pretrained',return_value=NS(model_type='fixture')), \
                patch.object(verify.AutoModelForCausalLM,'from_config',return_value=Reference()),patch.object(verify,'no_init_weights',return_value=nullcontext()), \
                patch.object(verify,'reference_weights',return_value=({},[],0)),patch.object(verify,'make_executor',side_effect=Executor), \
                patch.object(verify,'device_name',return_value='fixture'),patch.object(verify,'empty_cache'),patch.object(verify.torch,'tensor',side_effect=cpu_tensor):
            verify.main()
    finally:
        report=json.loads(output.read_text()) if output.exists() else {}
        print('report_passed='+str(report.get('passed'))+' identity='+str(report.get('workload_sha256')),file=sys.stderr)
'''


class GgufReferenceGateTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)

    def test_invalid_layout_and_numerical_evidence_fail(self):
        for optimized in (False,True):
            for case in ('backend','count','abi','overhead','context','numerical'):
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertNotIn('report_passed=True',result.stderr)
                    if case=='numerical':self.assertIn('report_passed=False',result.stderr)

    def test_valid_report_has_exact_workload_identity(self):
        from dllm.token_workload import token_workload_digest
        expected=token_workload_digest([[1]],2)
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('report_passed=True identity='+expected,result.stderr)
