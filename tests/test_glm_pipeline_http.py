import json
import os
import secrets
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor
from scripts import serve_cpu_reference as command
from tests import test_serve_cpu_reference as commands
from damendaemon.glm_model import GlmModelState
from damendaemon.glm_pipeline_generation import PipelineToolReferenceGenerator


class GlmPipelineHttpTests(unittest.TestCase):
    setUp=commands.ReferenceCommandTests.setUp
    snapshot=commands.ReferenceCommandTests.snapshot
    args=commands.ReferenceCommandTests.args
    event=commands.ReferenceCommandTests.event

    def test_actual_subprocess_json_sse_queue_status_drain_and_shutdown(self):
        path=self.snapshot('glm');results=[]
        for cuts in ([],['--layer-splits','2']):
            key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
            process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','glm',str(path),
                '--context','128','--port','0','--queue-size','3',*cuts],stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,env=self.env|{'DAMENDAEMON_API_KEY':key,'DAMENDAEMON_ADMIN_API_KEY':admin})
            try:
                ready=self.event(process);base=ready['url']
                if cuts:
                    self.assertEqual(ready['execution'],'local_cpu_pipeline')
                    self.assertEqual(ready['stage_layers'],[[0,2],[2,3]])
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                admin_headers=headers|{'Authorization':'Bearer '+admin}
                with self.assertRaises(HTTPError) as denied:urlopen(base+'/v1/models/status',timeout=3)
                self.assertEqual(denied.exception.code,401)
                with urlopen(Request(base+'/v1/models/status',headers=admin_headers),timeout=3) as response:
                    status=json.load(response)
                if cuts:self.assertEqual(status['stage_layers'],[[0,2],[2,3]])
                body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=2)
                def request():
                    with urlopen(Request(base+'/v1/chat/completions',json.dumps(body).encode(),headers),timeout=10) as response:
                        return json.load(response)
                with ThreadPoolExecutor(3) as pool:responses=list(pool.map(lambda _:request(),range(3)))
                results.append([(r['choices'][0]['message'],r['usage']) for r in responses])
                streamed=body|dict(stream=True,stream_options={'include_usage':True})
                with urlopen(Request(base+'/v1/chat/completions',json.dumps(streamed).encode(),headers),timeout=10) as response:
                    stream=response.read().decode()
                self.assertIn('data: [DONE]',stream)
                if cuts:self.assertIn('local_cpu_pipeline',stream)
                with urlopen(Request(base+'/v1/scheduler/drain',b'{}',admin_headers),timeout=3) as response:
                    self.assertTrue(json.load(response)['drained'])
                with self.assertRaises(HTTPError):request()
                process.send_signal(signal.SIGTERM);output,error=process.communicate(timeout=10)
                self.assertEqual(process.returncode,0,error);self.assertEqual(json.loads(output)['event'],'stopped')
                self.assertNotIn(key,output+error);self.assertNotIn(admin,output+error)
            finally:
                if process.poll() is None:process.kill();process.communicate()
                process.stdout.close();process.stderr.close()
        self.assertEqual(results[0],results[1])

    def test_invalid_family_or_cuts_precede_payload(self):
        path=self.snapshot('glm')
        with patch.dict(os.environ,self.env,clear=True),patch('damendaemon.glm_reference_loader._payload',side_effect=AssertionError('payload')):
            for family,cuts in (('flash',['4']),('deepseek',['2']),('glm',['1']),('glm',['2','2'])):
                with self.subTest(family=family,cuts=cuts),self.assertRaises(ValueError):
                    command.prepare(self.args(family,path,'--layer-splits',*cuts))

    def test_retained_stage_allocation_cleanup_is_observable_and_retryable(self):
        path=self.snapshot('glm')
        with patch.dict(os.environ,self.env,clear=True):runtime,_=command.prepare(self.args('glm',path,'--layer-splits','2'))
        self.addCleanup(runtime.close)
        self.assertIs(type(runtime.runner),PipelineToolReferenceGenerator)
        pipeline=runtime.runner.model;last=pipeline.stages[-1]
        with patch.object(last.blocks[0],'new_state',side_effect=MemoryError('allocation')),patch.object(GlmModelState,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(MemoryError,'allocation'):
                list(runtime.generate('model',[dict(role='user',content='Hi')],max_tokens=1))
            self.assertTrue(last.failed_cleanups)
            self.assertTrue(runtime.cleanup_pending())
            self.assertTrue(runtime.manage('status')['cleanup_pending'])
            with self.assertRaisesRegex(RuntimeError,'release'):runtime.close()
            self.assertFalse(runtime.closed)
        runtime.close();self.assertFalse(last.failed_cleanups);self.assertTrue(runtime.closed)

    def test_abandoned_or_failed_request_releases_pipeline_before_reuse(self):
        path=self.snapshot('glm')
        with patch.dict(os.environ,self.env,clear=True):runtime,_=command.prepare(self.args('glm',path,'--layer-splits','2'))
        self.addCleanup(runtime.close)
        pipeline=runtime.runner.model;states=[];create=pipeline.new_state
        def observe(*args,**kwargs):
            state=create(*args,**kwargs);states.append(state);return state
        messages=[dict(role='user',content='Hi')]
        with patch.object(pipeline,'new_state',observe):
            stream=runtime.generate('model',messages,max_tokens=5);next(stream);stream.close()
            self.assertTrue(states[-1].closed)
            with patch.object(pipeline.stages[-1],'forward',side_effect=RuntimeError('later stage')):
                with self.assertRaisesRegex(RuntimeError,'later stage'):list(runtime.generate('model',messages,max_tokens=2))
            self.assertTrue(states[-1].closed)
            self.assertTrue(list(runtime.generate('model',messages,max_tokens=1)))
            self.assertTrue(states[-1].closed)
