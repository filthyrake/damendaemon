"""Generated HTTP qualification with no launched server or device access."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import hashlib,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from scripts import verify_gguf_generated_http as verify
case=sys.argv[1];processes=[]
class Process:
    closed=False
    def poll(self):return None
    def terminate(self):self.closed=True
    def wait(self,**kwargs):return 0
    def kill(self):self.closed=True
def launch(*args,**kwargs):
    process=Process();processes.append(process);return process
def response(request,**kwargs):
    if isinstance(request,str):
        raw=io.BytesIO();raw.status=200;return raw
    body=json.loads(request.data)
    text='wrong' if case==('stream-text' if body['stream'] else 'text') else 'expected'
    if body['stream']:
        wire='data: '+json.dumps({'choices':[{'text':text}]})+'\n\n'
        if case!='no-done':wire+='data: [DONE]\n\n'
        return io.BytesIO(wire.encode())
    return io.BytesIO(json.dumps({'choices':[{'text':text}],'dllm_timing':{}}).encode())
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);model=root/'model.gguf';model.write_bytes(b'fixture')
    checksum=hashlib.sha256(model.read_bytes()).hexdigest()
    benchmark={'model_sha256':checksum,'prompts':{'1':[1]},'output_tokens':1,'runs':[{'engine':'llama.cpp','cases':[
        {'prompt_tokens':1,'concurrency':1,'rounds':[{'requests':[{'text':'expected'}]}]*4}]}]}
    reference={'passed':True,'sha256':checksum,'cases':[{'graph':False,'case':0,'prompt_tokens':1,'greedy_mismatches':[],'reference_tokens':[2]}]}
    reference['cases'].append(dict(reference['cases'][0],graph=True))
    reference['workload_sha256']=hashlib.sha256(b'{"output_tokens":1,"prompts":[[1]],"version":1}').hexdigest()
    if case=='trace-missing':reference['cases']=[]
    if case=='trace-duplicate':reference['cases'][1]=dict(reference['cases'][0])
    if case=='trace-bool-index':reference['cases'][0]['case']=False
    if case=='trace-bool-graph':reference['cases'][0]['graph']=0
    if case=='trace-short':reference['cases'][0]['reference_tokens']=[]
    if case=='trace-bool-token':reference['cases'][0]['reference_tokens']=[True]
    if case=='trace-vocab':reference['cases'][0]['reference_tokens']=[4]
    if case=='trace-mismatch':reference['cases'][0]['greedy_mismatches']=[0]
    if case=='trace-null-mismatch':reference['cases'][0]['greedy_mismatches']=None
    if case=='comparator-missing':benchmark['runs']=[]
    if case=='comparator-text':benchmark['runs'][0]['cases'][0]['rounds'][3]['requests'][0]['text']=None
    if case=='changed-token':benchmark['prompts']={'1':[3]}
    if case=='changed-output':benchmark['output_tokens']=2
    if case=='missing-identity':reference.pop('workload_sha256')
    if case=='prompt-order':
        reference['workload_sha256']=hashlib.sha256(b'{"output_tokens":1,"prompts":[[1],[3]],"version":1}').hexdigest()
        reference['cases'].append(dict(reference['cases'][0],case=1))
        benchmark['prompts']={'1':[3],'2':[1]}
    if case=='empty':benchmark['prompts']={}
    if case=='bad-reference':reference['passed']='yes'
    if case=='zero-output':benchmark['output_tokens']=0
    if case=='bool-output':benchmark['output_tokens']=True
    if case=='context':benchmark['prompts']={'512':[1]*512}
    if case=='bool-token':benchmark['prompts']={'1':[True]}
    if case=='many-prompts':benchmark['prompts']={str(i):[1] for i in range(65)}
    for name,value in [('benchmark',benchmark),('reference',reference)]:
        (root/f'{name}.json').write_text(json.dumps(value))
    output=root/'result.json'
    args=['verify',str(model),'--benchmark',str(root/'benchmark.json'),'--reference',str(root/'reference.json'),'--output',str(output)]
    try:
        with patch('sys.argv',args),patch.object(verify,'plan',return_value=NS(tokenizer_json={},sentencepiece_prefix=None,config={'vocab_size':4})), \
                patch.object(verify,'restore_tokenizer',return_value=NS(decode=lambda *args,**kwargs:'expected')), \
                patch.object(verify.subprocess,'Popen',side_effect=launch),patch.object(verify,'urlopen',side_effect=response):
            verify.main()
    finally:
        passed=json.loads(output.read_text())['passed'] if output.exists() else None
        print('launched='+str(len(processes))+' cleaned='+str(all(p.closed for p in processes))+' passed='+str(passed),file=sys.stderr)
'''


class GgufGeneratedHttpVerifierTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)

    def test_invalid_inputs_refuse_launch_and_bad_outputs_fail(self):
        before=('trace-missing','trace-duplicate','trace-bool-index','trace-bool-graph','trace-short','trace-bool-token','trace-vocab','trace-mismatch','trace-null-mismatch','comparator-missing','comparator-text','changed-token','changed-output','missing-identity','prompt-order','empty','bad-reference','zero-output','bool-output','context','bool-token','many-prompts')
        for optimized in (False,True):
            for case in (*before,'text','stream-text','no-done'):
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertNotIn('passed=True',result.stderr)
                    self.assertIn('cleaned=True',result.stderr)
                    if case in before:self.assertIn('launched=0',result.stderr)

    def test_valid_control_in_both_modes(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('launched=1 cleaned=True passed=True',result.stderr)
