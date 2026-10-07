import json
import os
import secrets
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.glm_flash_pipeline_generation import FlashPipelineToolReferenceGenerator
from dllm.glm_flash_model import FlashModelState
from scripts import serve_cpu_reference as command
from tests import test_serve_cpu_reference as fixtures


class FlashPipelineHttpTests(unittest.TestCase):
    setUp=fixtures.ReferenceCommandTests.setUp
    snapshot=fixtures.ReferenceCommandTests.snapshot
    args=fixtures.ReferenceCommandTests.args
    event=fixtures.ReferenceCommandTests.event

    def test_actual_subprocess_json_sse_queue_status_drain_and_signal_match_default(self):
        path=self.snapshot('flash');results=[]
        for cuts in ([],['--layer-splits','1','2']):
            key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
            process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','flash',str(path),
                '--context','128','--port','0','--queue-size','3',*cuts],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,env=self.env|{'DLLM_API_KEY':key,'DLLM_ADMIN_API_KEY':admin})
            try:
                ready=self.event(process);base=ready['url']
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                admin_headers=headers|{'Authorization':'Bearer '+admin}
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
                self.assertEqual(status['model_type'],'glm5_next')
                if cuts:self.assertEqual(status['execution'],'local_cpu_pipeline')
                with urlopen(Request(base+'/v1/scheduler/drain',b'{}',admin_headers),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
                process.send_signal(signal.SIGTERM);output,error=process.communicate(timeout=10)
                self.assertEqual(process.returncode,0,error);self.assertEqual(json.loads(output)['event'],'stopped')
                self.assertNotIn(key,output+error);self.assertNotIn(admin,output+error)
            finally:
                if process.poll() is None:process.kill();process.communicate()
                process.stdout.close();process.stderr.close()
        self.assertEqual(results[0],results[1])

    def test_images_deepseek_and_bad_cuts_fail_before_payload(self):
        path=self.snapshot('flash')
        with patch.dict(os.environ,self.env,clear=True),patch('dllm.glm_flash_reference_loader._payload',side_effect=AssertionError('payload')):
            for family,options in (('deepseek',['--layer-splits','1']),('flash',['--layer-splits','3']),
                                   ('glm',['--layer-splits','1','--flash-images'])):
                with self.assertRaises(ValueError):command.prepare(self.args(family,path,*options))

    def test_pipeline_cleanup_failure_remains_observable_and_retryable(self):
        path=self.snapshot('flash')
        with patch.dict(os.environ,self.env,clear=True):runtime,_=command.prepare(self.args('flash',path,'--layer-splits','1'))
        self.addCleanup(runtime.close);self.assertIs(type(runtime.runner),FlashPipelineToolReferenceGenerator)
        model=runtime.runner.model;last=model.stages[-1]
        with patch.object(last.blocks[0],'new_state',side_effect=MemoryError('allocation')),patch.object(FlashModelState,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaises(MemoryError):list(runtime.generate('model',[dict(role='user',content='Hi')],1))
            self.assertTrue(runtime.cleanup_pending());self.assertTrue(runtime.manage('status')['cleanup_pending'])
            with self.assertRaisesRegex(RuntimeError,'cleanup'):runtime.close()
        runtime.close();self.assertTrue(runtime.closed)
