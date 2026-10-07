import contextlib
import io
import json
import os
from pathlib import Path
import secrets
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from scripts import serve_cpu_reference as command
from tests import test_glm_generation as glm
from tests import test_glm_reference_loader as glm_loader
from tests import test_glm_flash_generation as flash
from tests import test_deepseek_generation as deepseek
from tests import test_deepseek_reference_loader as deepseek_loader


class ReferenceCommandTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.env={k:v for k,v in os.environ.items() if not k.startswith('DLLM_')}
        self.env['OMP_NUM_THREADS']='1';self.env['MKL_NUM_THREADS']='1'

    def snapshot(self,family):
        path=self.path/family;path.mkdir(exist_ok=True)
        if family=='deepseek':
            text=deepseek.text_fixture(path);deepseek_loader.fixture(path,vocab_size=text.vocab_size)
            name='head.weight'
        else:
            text=(glm if family=='glm' else flash).text_fixture(path)
            if family=='glm':glm_loader.fixture(path,vocab_size=text.vocab_size)
            else:flash.snapshot(path,text)
            name='lm_head.weight'
        # Keep real packed arithmetic while removing random untrained control output.
        mapping=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        shard=path/mapping[name]
        data=bytearray(shard.read_bytes());length=struct.unpack('<Q',data[:8])[0]
        header=json.loads(data[8:8+length]);start,end=header[name]['data_offsets']
        data[8+length+start:8+length+end]=bytes(end-start);shard.write_bytes(data)
        return path

    def args(self,family,path,*extra):
        return command.parser_for_command().parse_args([family,str(path),'--context','128','--port','0',*extra])

    def event(self,process):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            self.assertTrue(selector.select(20),'command did not announce readiness')
        line=process.stdout.readline()
        if not line:self.fail('command ended before readiness: '+process.stderr.read())
        return json.loads(line)

    def test_actual_snapshots_authenticated_http_and_signal_shutdown(self):
        for family in ('glm','flash','deepseek'):
            with self.subTest(family=family):
                path=self.snapshot(family);key=secrets.token_urlsafe(32)
                process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference',family,str(path),
                    '--context','128','--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                    env=self.env|{'DLLM_API_KEY':key})
                try:
                    ready=self.event(process);self.assertEqual(ready['event'],'http_ready')
                    self.assertFalse(ready['official_model_qualified']);self.assertEqual(ready['scope'],'cpu_reference')
                    with self.assertRaises(HTTPError) as denied:urlopen(ready['url']+'/v1/models',timeout=3)
                    self.assertEqual(denied.exception.code,401)
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                    with urlopen(Request(ready['url']+'/health',headers=headers),timeout=3) as response:
                        self.assertEqual(response.status,200)
                    for stream in (False,True):
                        body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=1,stream=stream)
                        if stream:body['stream_options']={'include_usage':True}
                        with urlopen(Request(ready['url']+'/v1/chat/completions',json.dumps(body).encode(),headers),timeout=10) as response:
                            payload=response.read().decode();self.assertEqual(response.status,200)
                        if stream:self.assertIn('data: [DONE]',payload);self.assertIn('cpu_reference',payload)
                        else:
                            result=json.loads(payload);self.assertEqual(result['usage']['completion_tokens'],1)
                            self.assertEqual(result['dllm_timing']['source_tier'],'cpu_reference')
                    process.send_signal(signal.SIGTERM)
                    output,error=process.communicate(timeout=10)
                    self.assertEqual(process.returncode,0,error);self.assertEqual(json.loads(output)['event'],'stopped')
                    self.assertNotIn(key,error+output)
                finally:
                    if process.poll() is None:process.kill();process.communicate()
                    process.stdout.close();process.stderr.close()

    def test_preflight_rejects_policy_family_metadata_and_budgets_before_payload(self):
        path=self.snapshot('glm')
        with patch.dict(os.environ,self.env,clear=True),patch('dllm.glm_reference_loader.load_reference_model',side_effect=AssertionError('payload')) as load:
            for extra in (('--alias',''),('--context','1048577'),('--cache-mib','2049'),('--port','65536'),('--connections','0')):
                with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    command.main(['glm',str(path),*extra])
            with patch.dict(os.environ,{'DLLM_API_KEY':'bad'}),self.assertRaises(ValueError):command.prepare(self.args('glm',path))
            with self.assertRaises(ValueError):command.prepare(self.args('deepseek',path))
            with self.assertRaises(ValueError):command.prepare(self.args('deepseek',path,'--output-mib','4'))
            config=json.loads((path/'config.json').read_text());config['max_position_embeddings']=2
            (path/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):command.prepare(self.args('glm',path))
            load.assert_not_called()

    def test_bind_failure_and_active_shutdown_retain_cleanup_ownership(self):
        path=self.snapshot('glm')
        with patch.dict(os.environ,self.env,clear=True):runtime,auth=command.prepare(self.args('glm',path))
        with patch.object(command,'BoundedHTTPServer',side_effect=OSError('bind')),self.assertRaisesRegex(OSError,'bind'):
            command.serve(runtime,auth,self.args('glm',path))
        self.assertTrue(runtime.closed)
        with patch.dict(os.environ,self.env,clear=True):runtime,auth=command.prepare(self.args('glm',path))
        started=threading.Event();finish=threading.Event();observed=[]
        def request():
            with runtime.lock:
                runtime.active=True;started.set();finish.wait(3)
                observed.append(runtime.stopping.is_set());runtime.active=False
        worker=threading.Thread(target=request);worker.start();self.assertTrue(started.wait(3))
        class Server:
            server_port=1
            def handle_request(self):raise RuntimeError('stop')
            def server_close(self):finish.set();worker.join(3)
        with patch.object(command,'BoundedHTTPServer',return_value=Server()),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'stop'):
            command.serve(runtime,auth,self.args('glm',path))
        self.assertEqual(observed,[True]);self.assertTrue(runtime.closed);self.assertFalse(worker.is_alive())

    def test_failed_deepseek_prefill_cleanup_blocks_health_and_shutdown_until_retry(self):
        from dllm.floating_model import ModelState
        path=self.snapshot('deepseek')
        with patch.dict(os.environ,self.env,clear=True):runtime,_=command.prepare(self.args('deepseek',path))
        model=runtime.runner.model
        with patch.object(model,'_logits',side_effect=RuntimeError('primary prefill')),patch.object(ModelState,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'primary prefill') as failure:
                list(runtime.generate('model',[dict(role='user',content='Hi')],max_tokens=1))
            self.assertTrue(failure.exception.__notes__)
            self.assertTrue(runtime.cleanup_pending());self.assertTrue(model.failed_prefills)
            with self.assertRaisesRegex(RuntimeError,'release'):runtime.close()
            self.assertIs(runtime.runner.model,model);self.assertFalse(runtime.closed)
        runtime.close()
        self.assertFalse(model.failed_prefills);self.assertTrue(runtime.closed)
