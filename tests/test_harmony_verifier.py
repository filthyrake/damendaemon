"""Run the real verifier with in-memory HTTP, without a GPU or server."""
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from urllib.error import HTTPError
from scripts import verify_harmony_tools as verify
case=sys.argv[1];responses=[];counter=[0]
runtime=NS(generate=Mock(return_value=iter([{'token':1},{'finish_reason':'tool_calls'}])),close=Mock())
server=NS(server_port=1,serve_forever=Mock(),shutdown=Mock(),server_close=Mock())
def response(request,**kwargs):
    body=json.loads(request.data);counter[0]+=1
    if body.get('tool_choice') in ('required',) or isinstance(body.get('tool_choice'),dict) or body['tools'][0]['function'].get('strict'):
        raw=io.BytesIO(b'{}');responses.append(raw)
        raise HTTPError(request.full_url,500 if case=='wrong-status' else 400,'fixture',{},raw)
    usage={'prompt_tokens':10,'completion_tokens':10}
    message={'content':'','reasoning_content':'thinking'};finish='tool_calls'
    if body.get('tool_choice')=='none':
        message={'content':'5' if case=='none-choice' else '4'};finish='stop'
    elif len(body['messages'])>1:
        message={'content':'missing' if case=='history' else '19 degrees'};finish='stop'
    elif body['max_tokens']<128:
        message={'content':'partial','reasoning_content':''};finish='stop' if case=='cap' else 'length'
        usage['completion_tokens']=body['max_tokens']
    else:
        message['tool_calls']=[{'id':'same' if case=='shared-id' else str(counter[0]),'type':'function',
            'function':{'name':'wrong' if case=='wrong-tool' else 'weather',
                        'arguments':json.dumps({'city':'Rome' if case=='wrong-args' else 'Paris'})}}]
    payload={'choices':[{'message':message,'finish_reason':finish}],'usage':usage}
    if body.get('stream'):
        delta=dict(message)
        if 'tool_calls' in delta:
            delta['tool_calls']=[dict(c,index=1 if case=='index' else i) for i,c in enumerate(delta['tool_calls'])]
        if case=='diverge':delta['content']='different'
        chunks=[{'choices':[{'delta':delta,'finish_reason':finish}]}]
        if case=='stream-error':chunks.insert(0,{'error':'injected','choices':[]})
        chunks.append({'choices':[],'usage':usage})
        wire=''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)
        if case!='no-done':wire+='data: [DONE]'+('' if case=='truncated-done' else '\n\n')
        raw=io.BytesIO(wire.encode())
    else:raw=io.BytesIO(json.dumps(payload).encode())
    responses.append(raw);return raw
with tempfile.TemporaryDirectory() as tmp:
    output=Path(tmp)/'report.json'
    try:
        with patch('sys.argv',['verify','unused','--output',str(output)]), \
                patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'}), \
                patch.object(verify,'BatchedRuntime',return_value=runtime), \
                patch.object(verify,'BoundedHTTPServer',return_value=server), \
                patch.object(verify,'torch',NS(set_num_threads=Mock(),cuda=NS(get_device_name=lambda:'fake'))), \
                patch.object(verify,'urlopen',side_effect=response):
            verify.main()
    finally:
        print('published='+str(output.exists())+' cleaned='+str(runtime.close.called and server.shutdown.called and server.server_close.called)+' responses_closed='+str(all(r.closed for r in responses)),file=sys.stderr)
'''


class HarmonyVerifierTests(unittest.TestCase):
    def run_case(self,case,optimized):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)

    def test_invalid_qualification_fails_without_publishing(self):
        for optimized in (False,True):
            for case in ('wrong-tool','wrong-args','index','diverge','history','cap','none-choice',
                         'shared-id','wrong-status','stream-error','no-done','truncated-done'):
                with self.subTest(case=case,optimized=optimized):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertIn('published=False cleaned=True responses_closed=True',result.stderr)

    def test_valid_qualification_runs_in_both_modes(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('published=True cleaned=True responses_closed=True',result.stderr)
