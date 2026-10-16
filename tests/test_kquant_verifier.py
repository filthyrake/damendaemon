"""Qualification harness checks using tiny CPU tensors and in-memory HTTP."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io,json,sys,threading
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from scripts import verify_kquant_runtime as verify
case=sys.argv[1];tier=['disk'];responses=[]
model=NS(bytes=16,raw_tokens=lambda text:[1,2],reference=False)
reference=NS(bytes=32,reference=True)
class Executor:
    kernel_backend='fixture';packed_workspace_bytes=0
    def __init__(self,model,**kwargs):self.reference=model.reference
    def scores(self,rows):
        values=[0.,2.,1.]
        if not self.reference:
            if case=='numerical':values=[0.,4.,1.]
            if case=='greedy':values=[0.,1.,2.]
            if case=='nonfinite':values=[0.,float('nan'),1.]
        return verify.torch.tensor([values]*rows)
    def prefill_batch(self,prompts,slots):return self.scores(len(prompts))
    def decode_batch(self,tokens,slots):return self.scores(len(tokens)),None
    def prefill_chunk(self,*args,**kwargs):return self.scores(1)
runtime=NS(runtime=NS(lock=threading.Lock(),host={}),close=Mock())
server=NS(server_port=1,serve_forever=Mock(),shutdown=Mock(),server_close=Mock())
def response(request,**kwargs):
    body=json.loads(request.data);route=request.full_url.rsplit('/',1)[-1]
    if route=='unload':tier[0]='disk' if body.get('release_host') else 'host'
    if route in ('register','unload'):payload={}
    else:
        text='different' if case=='chat' and body['model']=='expanded' else 'answer'
        if case=='reload-text' and tier[0]=='host':text='changed'
        usage={'prompt_tokens':len(body.get('prompt',[])),'completion_tokens':body['max_tokens']}
        if case=='work-count' and 'prompt' in body:usage['completion_tokens']=1
        if case=='trial-text' and 'prompt' in body and body['model']=='expanded':text='changed'
        payload={'choices':[{'text':text,'message':{'content':text}}],'usage':usage,
                 'dllm_timing':{'source_tier':'wrong' if case=='reload-tier' and tier[0]=='host' else tier[0]}}
    if body.get('stream'):
        text='changed' if case=='stream-text' else 'answer'
        wire='data: '+json.dumps({'choices':[{'delta':{'content':text}}]})+'\n\n'
        if case!='no-done':wire+='data: [DONE]\n\n'
        raw=io.BytesIO(wire.encode())
    else:raw=io.BytesIO(json.dumps(payload).encode())
    responses.append(raw);return raw
try:
    with patch('sys.argv',['verify','unused']),patch.object(verify,'require_gpu'), \
            patch.object(verify,'device_name',return_value='fixture'),patch.object(verify,'empty_cache'), \
            patch.object(verify,'GgufWeights',return_value=model),patch.object(verify,'expanded',return_value=reference), \
            patch.object(verify,'make_executor',side_effect=Executor),patch.object(verify,'BatchedRuntime',return_value=runtime), \
            patch.object(verify,'BoundedHTTPServer',return_value=server),patch.object(verify,'urlopen',side_effect=response):
        verify.main()
finally:
    print('closed='+str(runtime.close.called)+' responses_closed='+str(all(r.closed for r in responses)),file=sys.stderr)
'''


class KquantVerifierTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)

    def test_bad_evidence_never_publishes_pass(self):
        for optimized in (False,True):
            for case in ('numerical','greedy','nonfinite','chat','stream-text','no-done',
                         'work-count','trial-text','reload-text','reload-tier'):
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertNotIn('"passed": true',result.stdout)
                    self.assertIn('responses_closed=True',result.stderr)
                    if case not in ('numerical','greedy','nonfinite'):
                        self.assertIn('closed=True',result.stderr)

    def test_valid_control_in_both_modes(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"passed": true',result.stdout)
            self.assertIn('closed=True responses_closed=True',result.stderr)
