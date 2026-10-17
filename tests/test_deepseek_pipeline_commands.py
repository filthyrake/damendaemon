import contextlib
import io
import json
import os
import secrets
import signal
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from scripts import run_deepseek_reference as generate,serve_cpu_reference as serve
from tests import test_serve_cpu_reference as fixtures


class DeepSeekPipelineCommandTests(unittest.TestCase):
    setUp=fixtures.ReferenceCommandTests.setUp
    snapshot=fixtures.ReferenceCommandTests.snapshot
    event=fixtures.ReferenceCommandTests.event
    args=fixtures.ReferenceCommandTests.args

    def test_native_subprocess_single_equal_ragged_and_tool_requests_match_default(self):
        path=self.snapshot('deepseek');message=[dict(role='user',content='Hi')]
        equal=self.path/'equal.json';equal.write_text(json.dumps([message,message]))
        ragged=self.path/'ragged.json';ragged.write_text(json.dumps([message,[dict(role='user',content='Longer request')]]))
        tools=self.path/'tools.json';tools.write_text(json.dumps([dict(type='function',function=dict(name='lookup',parameters=dict(type='object',properties={}))) ]))
        for options in (['--prompt','Hi'],['--batch-messages',str(equal)],['--batch-messages',str(ragged),'--ragged-batch'],['--prompt','Hi','--tools',str(tools)]):
            results=[]
            for cuts in ([],['--layer-splits','1','2']):
                result=subprocess.run([sys.executable,'-m','scripts.run_deepseek_reference',str(path),'--context','2048','--max-new-tokens','2',*cuts,*options],env=self.env,capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr);events=[json.loads(line) for line in result.stdout.splitlines()]
                self.assertFalse(events[0]['official_model_qualified'])
                if cuts:self.assertEqual(events[0]['stage_layers'],[[0,1],[1,2],[2,3]])
                results.append(events[1:])
            self.assertEqual(*results)

    def test_bad_cuts_context_dtype_and_tools_fail_before_payload(self):
        path=self.snapshot('deepseek');tools=self.path/'bad.json';tools.write_text('[{}]')
        with patch('damendaemon.deepseek_reference_loader._payload',side_effect=AssertionError('payload')):
            for options in (['--layer-splits','3'],['--layer-splits','1','1'],['--layer-splits','1','--dtype','float16'],['--layer-splits','1','--context','1'],['--layer-splits','1','--tools',str(tools)],['--output-mib','1']):
                with self.subTest(options=options),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    generate.main([str(path),'--prompt','Hi','--max-new-tokens','1',*options])
            with patch.dict(os.environ,self.env,clear=True):
                for options in (['--layer-splits','3'],['--layer-splits','1','--workspace-mib','1'],['--layer-splits','1','--output-mib','0']):
                    with self.assertRaises((ValueError,SystemExit)),contextlib.redirect_stderr(io.StringIO()):serve.prepare(self.args('deepseek',path,*options))

    def test_http_json_sse_concurrent_queue_status_drain_and_signal_match_default(self):
        path=self.snapshot('deepseek');results=[]
        for cuts in ([],['--layer-splits','1','2']):
            key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
            process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','deepseek',str(path),'--context','128','--port','0','--queue-size','3',*cuts],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.env|{'DAMENDAEMON_API_KEY':key,'DAMENDAEMON_ADMIN_API_KEY':admin})
            try:
                ready=self.event(process);base=ready['url'];headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'};admin_headers=headers|{'Authorization':'Bearer '+admin}
                with self.assertRaises(HTTPError) as denied:urlopen(base+'/v1/models/status',timeout=3)
                self.assertEqual(denied.exception.code,401);denied.exception.close()
                if cuts:self.assertEqual(ready['stage_layers'],[[0,1],[1,2],[2,3]])
                body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=2)
                def request():
                    with urlopen(Request(base+'/v1/chat/completions',json.dumps(body).encode(),headers),timeout=10) as response:return json.load(response)
                with ThreadPoolExecutor(3) as pool:responses=list(pool.map(lambda _:request(),range(3)))
                results.append([(r['choices'][0]['message'],r['usage']) for r in responses])
                with urlopen(Request(base+'/v1/chat/completions',json.dumps(body|dict(stream=True,stream_options={'include_usage':True})).encode(),headers),timeout=10) as response:wire=response.read().decode()
                self.assertIn('data: [DONE]',wire)
                if cuts:self.assertIn('local_cpu_pipeline',wire)
                with urlopen(Request(base+'/v1/models/status',headers=admin_headers),timeout=3) as response:status=json.load(response)
                self.assertEqual(status['model_type'],'deepseek_v4')
                if cuts:self.assertEqual(status['execution'],'local_cpu_pipeline')
                with urlopen(Request(base+'/v1/scheduler/drain',b'{}',admin_headers),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
                process.send_signal(signal.SIGTERM);output,error=process.communicate(timeout=10)
                self.assertEqual(process.returncode,0,error);self.assertEqual(json.loads(output)['event'],'stopped');self.assertNotIn(key,output+error)
            finally:
                if process.poll() is None:process.kill();process.communicate()
                process.stdout.close();process.stderr.close()
        self.assertEqual(*results)

    def test_partial_prefill_cleanup_remains_visible_until_retry(self):
        from damendaemon.deepseek_layer_stage import DeepSeekStageState
        path=self.snapshot('deepseek')
        with patch.dict(os.environ,self.env,clear=True):runtime,_=serve.prepare(self.args('deepseek',path,'--layer-splits','1'))
        self.addCleanup(runtime.close);model=runtime.runner.model
        with patch.object(model.stages[-1],'prefill',side_effect=MemoryError('later stage')),patch.object(DeepSeekStageState,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaises(MemoryError):list(runtime.generate('model',[dict(role='user',content='Hi')],1))
            self.assertTrue(runtime.cleanup_pending());self.assertTrue(runtime.manage('status')['cleanup_pending'])
            with self.assertRaisesRegex(RuntimeError,'cleanup'):runtime.close()
        runtime.close();self.assertTrue(runtime.closed)
