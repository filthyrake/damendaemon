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
from scripts import run_glm_flash_image_reference as command
from tests.test_glm_flash_inline_images import snapshot,messages
from tests.test_glm_tools import TOOLS
from tests import test_serve_cpu_reference as fixtures


class FlashPipelineImageCommandTests(unittest.TestCase):
    setUp=fixtures.ReferenceCommandTests.setUp
    event=fixtures.ReferenceCommandTests.event

    def test_native_command_with_images_and_tools_matches_whole_model(self):
        snapshot(self.path);source=self.path/'messages.json';source.write_text(json.dumps(messages()))
        tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        for extra in ([],['--tools',str(tools)]):
            results=[]
            for cuts in ([],['--layer-splits','1','2']):
                process=subprocess.run([sys.executable,'-m','scripts.run_glm_flash_image_reference',str(self.path),
                    '--messages',str(source),'--context','2048','--max-new-tokens','2',*extra,*cuts],
                    capture_output=True,text=True,env=self.env,timeout=20)
                self.assertEqual(process.returncode,0,process.stderr)
                events=[json.loads(line) for line in process.stdout.splitlines()]
                if cuts:self.assertEqual(events[0]['stage_layers'],[[0,1],[1,2],[2,3]])
                results.append(events[1:])
            self.assertEqual(*results)
        with patch('damendaemon.glm_flash_reference_loader._payload',side_effect=AssertionError('payload')),contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--layer-splits','3'],['--context','1','--layer-splits','1']):
                with self.assertRaises(SystemExit) as error:command.main([str(self.path),'--messages',str(source),*extra])
                self.assertEqual(error.exception.code,2)

    def test_actual_queued_image_http_and_tools_match_whole_model(self):
        snapshot(self.path);results=[]
        for cuts in ([],['--layer-splits','1','2']):
            key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
            process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','flash',str(self.path),
                '--flash-images','--context','2048','--port','0','--queue-size','3',*cuts],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                env=self.env|{'DAMENDAEMON_API_KEY':key,'DAMENDAEMON_ADMIN_API_KEY':admin})
            try:
                ready=self.event(process);base=ready['url'];headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                if cuts:self.assertEqual(ready['stage_layers'],[[0,1],[1,2],[2,3]])
                body=dict(model='model',messages=messages(),max_tokens=2)
                def request(extra):
                    with urlopen(Request(base+'/v1/chat/completions',json.dumps(body|extra).encode(),headers),timeout=15) as response:return json.load(response)
                with ThreadPoolExecutor(3) as pool:responses=list(pool.map(request,({},dict(tools=TOOLS),{})))
                results.append([(r['choices'][0]['message'],r['usage']) for r in responses])
                with urlopen(Request(base+'/v1/chat/completions',json.dumps(body|dict(stream=True,stream_options={"include_usage":True})).encode(),headers),timeout=15) as response:wire=response.read().decode()
                self.assertIn('data: [DONE]',wire)
                if cuts:self.assertIn('local_cpu_pipeline',wire)
                admin_headers=headers|{'Authorization':'Bearer '+admin}
                with urlopen(Request(base+'/v1/models/status',headers=admin_headers),timeout=3) as response:status=json.load(response)
                self.assertEqual(status['modality'],'still_image');self.assertEqual(status['model_type'],'glm5_next')
                with urlopen(Request(base+'/v1/scheduler/drain',b'{}',admin_headers),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
                process.send_signal(signal.SIGTERM);output,error=process.communicate(timeout=10)
                self.assertEqual(process.returncode,0,error);self.assertEqual(json.loads(output)['event'],'stopped')
                self.assertNotIn(key,output+error);self.assertNotIn(admin,output+error)
            finally:
                if process.poll() is None:process.kill();process.communicate()
                process.stdout.close();process.stderr.close()
        self.assertEqual(*results)
