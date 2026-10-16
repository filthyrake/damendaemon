"""Actual verifier entry point with CPU logits and simulated HTTP/process ownership."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from scripts import verify_gguf_runtime as verify
case=sys.argv[1];responses=[];calls=[0];unloaded=[False]
def tensor(values):return verify.torch.tensor(values,device='cpu')
def model(reference):
    return NS(config={},quantization='fixture',reference=reference,eos=set(),
        chat_tokens=lambda messages:[1],raw_tokens=lambda text:[1],
        weights={'weight':tensor([2. if case=='weights' and not reference else 1.])},
        scales={'scale':tensor([2. if case=='scales' and not reference else 1.])})
source=model(False);reference=model(True)
class Executor:
    def __init__(self,model,use_graph=False):self.reference=model.reference;self.graph=use_graph
    def prefill(self,ids):
        bad=case=='numerical' and not self.reference
        # Both sources match within each run, but graph changes the generated trace.
        graph_change=case=='graph' and self.graph
        return tensor([[0.,3.,1.]] if not (bad or graph_change) else [[0.,1.,3.]])
    def decode(self,token):return self.prefill([]),None
process=NS(poll=Mock(return_value=None),send_signal=Mock(),wait=Mock(),kill=Mock())
def popen(*args,**kwargs):
    kwargs['stdout'].write('dllm listening on http://fixture\n');kwargs['stdout'].flush();return process
def response(request,**kwargs):
    body=json.loads(request.data);route=request.full_url.rsplit('/',1)[-1]
    if case in ('http-error','http-error-decode'):
        raw=io.BytesIO(b'failure' if case=='http-error' else bytes([255]));responses.append(raw)
        raise verify.HTTPError(request.full_url,500,'fixture',{},raw)
    if route=='unload':
        unloaded[0]=True
        payload={'host_cached_bytes':1 if case=='unload-bytes' else 0,
                 'models':[{'residency':'host' if case=='unload-residency' else 'disk'}]}
    elif route=='register':payload={}
    else:
        calls[0]+=1;n=calls[0];count=body['max_tokens'];text='1'*count
        if case=='http-text' and n==1:text='wrong'
        if case=='http-usage' and n==1:count-=1
        if case=='concurrent' and n in (6,7):text='wrong'
        if case=='recovery' and n==9:text='wrong'
        if case=='disk-text' and unloaded[0]:text='wrong'
        tier='disk' if unloaded[0] else 'host'
        if (case=='host-tier' and n==4) or (case=='disk-tier' and unloaded[0]):tier='wrong'
        payload={'choices':[{'text':text,'message':{'content':text}}],
                 'usage':{'prompt_tokens':1,'completion_tokens':count},'dllm_timing':{'source_tier':tier}}
    if body.get('stream'):
        text='wrong' if case=='stream-text' else '1'*body['max_tokens']
        parts=[text] if case=='cancel-done' and n==8 else [text[:1],text[1:]]
        wire=''.join('data: '+json.dumps({'choices':[{'delta':{'content':part},'text':part,
            'finish_reason':'stop' if case=='cancel-finished' and n==8 else None}]})+'\n\n' for part in parts)
        if case=='cancel-empty' and n==8:wire=''
        if case!='no-done':wire+='data: [DONE]\n\n'
        raw=io.BytesIO(wire.encode())
    else:raw=io.BytesIO(json.dumps(payload).encode())
    responses.append(raw);return raw
try:
    with patch('sys.argv',['verify','unused-gguf','unused-hf']), \
            patch.object(verify,'GgufWeights',return_value=source),patch.object(verify,'ModelWeights',return_value=reference), \
            patch.object(verify,'make_executor',side_effect=Executor),patch.object(verify,'device_name',return_value='fixture'), \
            patch.object(verify,'empty_cache'),patch.object(verify,'decode_output',side_effect=lambda model,tokens:''.join(map(str,tokens))), \
            patch.object(verify.subprocess,'Popen',side_effect=popen),patch.object(verify,'urlopen',side_effect=response):
        verify.main()
finally:
    print('stopped='+str(process.send_signal.called)+' responses_closed='+str(all(r.closed for r in responses)),file=sys.stderr)
'''


class GgufRuntimeGateTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)

    def test_bad_evidence_cannot_pass(self):
        for optimized in (False,True):
            for case in ('weights','scales','numerical','graph','http-text','http-usage',
                         'host-tier','stream-text','no-done','concurrent','recovery',
                         'unload-bytes','unload-residency','disk-text','disk-tier','cancel-done','cancel-finished','cancel-empty','http-error','http-error-decode'):
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertNotIn('"passed": true',result.stdout)
                    self.assertIn('responses_closed=True',result.stderr)
                    if case not in ('weights','scales','numerical','graph'):
                        self.assertIn('stopped=True',result.stderr)

    def test_valid_control(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"passed": true',result.stdout)
            self.assertIn('stopped=True responses_closed=True',result.stderr)
