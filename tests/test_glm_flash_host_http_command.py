from contextlib import redirect_stderr
import io
import json
import secrets
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.cpu_snapshot_identity import fingerprint
from scripts import run_glm_flash_host_topology as cli,run_cpu_host_topology as shared
from tests import test_glm_flash_native_host_command as native


class FlashHostHttpCommandTests(unittest.TestCase):
    setUp=native.FlashNativeHostCommandTests.setUp
    snapshot=native.FlashNativeHostCommandTests.snapshot
    launch=native.FlashNativeHostCommandTests.launch
    event=native.FlashNativeHostCommandTests.event
    workers=native.FlashNativeHostCommandTests.workers
    finish_workers=native.FlashNativeHostCommandTests.finish_workers

    def args(self,role,index=0,mode='fp8_reference',extra=()):
        return native.FlashNativeHostCommandTests.args(self,role,index,mode,extra)

    def service(self):
        key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
        self.environment.update(DLLM_API_KEY=key,DLLM_ADMIN_API_KEY=admin)
        identity=fingerprint(self.path).model_id;extra=['--verify-model-id','--model-id',identity]
        workers=self.workers('fp8_reference',extra)
        coordinator=self.launch(self.args('serve',extra=extra));ready=self.event(coordinator)
        self.assertEqual(ready['event'],'http_ready')
        self.assertTrue(ready['native_chat']);self.assertTrue(ready['model_identity_verified'])
        self.assertFalse(ready['queue_enabled']);self.assertFalse(ready['official_model_qualified'])
        self.assertEqual(ready['execution'],'cpu_host_topology')
        self.base=ready['url'];self.headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
        self.admin=self.headers|{'Authorization':'Bearer '+admin}
        return workers,coordinator,(key,admin)

    def post(self,**options):
        body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=3)|options
        return urlopen(Request(self.base+'/v1/chat/completions',json.dumps(body).encode(),self.headers),timeout=8)

    def test_actual_verified_http_json_sse_auth_heartbeat_and_idle_sigterm_stop(self):
        workers,coordinator,keys=self.service()
        with self.assertRaises(HTTPError) as error:urlopen(self.base+'/v1/models/status',timeout=3)
        self.assertEqual(error.exception.code,401);error.exception.close()
        with self.post() as response:first=json.load(response)
        with self.post() as response:second=json.load(response)
        self.assertEqual(first['choices'],second['choices']);self.assertEqual(first['usage'],second['usage'])
        with self.post(stream=True,stream_options={'include_usage':True}) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire);self.assertIn('cpu_host_topology',wire)
        with urlopen(Request(self.base+'/v1/models/status',headers=self.admin),timeout=3) as response:status=json.load(response)
        self.assertFalse(status['epoch_terminal']);self.assertTrue(status['model_identity_verified'])
        # Several authenticated requests allow the short idle heartbeat interval
        # to elapse; a subsequent request must still own a clean reusable epoch.
        with self.post() as response:self.assertEqual(response.status,200)
        coordinator.send_signal(signal.SIGTERM);output,error=coordinator.communicate(timeout=8)
        self.assertEqual(coordinator.returncode,0,error)
        stopped=json.loads(output);self.assertTrue(stopped['graceful_stop'])
        self.finish_workers(workers)
        for key in keys:self.assertNotIn(key,output+error)

    def test_bad_serving_options_fail_before_network(self):
        self.peers.write_text(json.dumps([dict(stage=i,address='127.0.0.1',port=1234+i,
            server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]))
        for extra in (['--queue-size','65'],['--queue-timeout','1'],['--idle-timeout','1'],['--stop-after-drain'],
                      ['--client-decode-turns'],['--interleaved-prefill'],['--tools','unused'],['--prompt','Hi'],
                      ['--heartbeat-interval','6'],['--http-port','65536']):
            with self.subTest(extra=extra),patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:cli.main(self.args('serve',extra=extra))
                self.assertEqual(error.exception.code,2)
        args=self.args('serve');args.remove('--native-chat')
        with patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(args)

    def test_peer_loss_stops_http_without_graceful_result(self):
        workers,coordinator,_=self.service();workers[-1].terminate()
        output,error=workers[-1].communicate(timeout=8);self.assertEqual(workers[-1].returncode,130,error)
        output,error=coordinator.communicate(timeout=8);self.assertEqual(coordinator.returncode,2,error)
        self.assertNotIn('graceful_stop',output)
        self.finish_workers(workers[:-1],2)

    def test_active_sigterm_cancels_sse_and_discards_epoch(self):
        workers,coordinator,_=self.service()
        with self.post(stream=True,max_tokens=1800) as response:
            prefix=[]
            for _ in range(20):
                line=response.readline().decode();prefix.append(line)
                if 'reasoning_content' in line:break
            else:self.fail('no active generation delta before signal')
            coordinator.send_signal(signal.SIGTERM)
            wire=''.join(prefix)+response.read().decode()
        self.assertNotIn('data: [DONE]',wire)
        output,error=coordinator.communicate(timeout=8)
        self.assertEqual(coordinator.returncode,130,error);self.assertNotIn('graceful_stop',output)
        self.finish_workers(workers,2)
