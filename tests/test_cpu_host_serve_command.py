import io
import json
import secrets
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from scripts import run_cpu_host_reference as cli
from tests import test_cpu_host_command as fixtures


class CpuHostServeCommandTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CpuHostCommandTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def args(self,port):
        args=self.fixture.args('complete',port);args[0]='serve'
        index=args.index('--batch-inputs');del args[index:index+2]
        return args+['--http-port','0','--heartbeat-interval','.1']

    def start_service(self,*,idle='1',environment=None,extra=(),worker_extra=()):
        worker,port=self.fixture.start_worker(extra=worker_extra)
        path=self.fixture.path/'http-service.jsonl';output=path.open('w')
        process=subprocess.Popen([sys.executable,'-m','scripts.run_cpu_host_reference',*self.args(port),'--idle-timeout',idle,*extra],
            stdout=output,stderr=subprocess.PIPE,text=True,env=environment or self.fixture.environment)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5);output.close()
        self.addCleanup(cleanup)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            events=[json.loads(line) for line in path.read_text().splitlines()]
            ready=next((event for event in events if event.get('event')=='http_ready'),None)
            if ready:return process,worker,ready,path
            if process.poll() is not None:self.fail(process.communicate()[1])
            time.sleep(.01)
        self.fail('CPU host HTTP service did not become ready')

    def test_two_process_http_sse_auth_and_idle_stop(self):
        environment=dict(self.fixture.environment,DLLM_API_KEY=secrets.token_urlsafe(32))
        service,worker,ready,path=self.start_service(environment=environment)
        base=ready['url'];self.assertTrue(base.startswith('http://127.0.0.1:'))
        self.assertFalse(ready['official_model_qualified'])
        payload=json.dumps({'model':'model','prompt':'hello world','max_tokens':3,'stream':True,
                            'stream_options':{'include_usage':True}}).encode()
        with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/v1/completions',data=payload,headers={'Content-Type':'application/json'}))
        self.assertEqual(error.exception.code,401)
        with urlopen(Request(base+'/v1/completions',data=payload,headers={'Content-Type':'application/json',
                'Authorization':'Bearer '+environment['DLLM_API_KEY']}),timeout=5) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire)
        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        self.assertEqual(events[-1]['usage']['completion_tokens'],3)
        self.assertEqual(events[-1]['dllm_timing']['kernel_backend'],'cpu-host-reference')
        _,error=service.communicate(timeout=6);worker_output,worker_error=worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error);self.assertEqual(worker.returncode,0,worker_error)
        self.assertEqual(json.loads(worker_output.splitlines()[-1])['event'],'stopped')
        self.assertEqual(json.loads(path.read_text().splitlines()[-1])['event'],'stopped')
        self.assertNotIn(environment['DLLM_API_KEY'],path.read_text()+error+worker_error)

    def test_verified_http_snapshot_ready_and_completion(self):
        from dllm.cpu_snapshot_identity import fingerprint
        identity=fingerprint(self.fixture.path).model_id
        extra=['--verify-model-id','--model-id',identity]
        service,worker,ready,_=self.start_service(extra=extra,worker_extra=extra)
        self.assertTrue(ready['model_identity_verified']);self.assertEqual(ready['source_model_id'],identity)
        payload=json.dumps(dict(model='model',prompt=[1,2],max_tokens=2)).encode()
        with urlopen(Request(ready['url']+'/v1/completions',payload,{'Content-Type':'application/json'}),timeout=5) as response:
            self.assertEqual(json.load(response)['usage']['completion_tokens'],2)
        _,error=service.communicate(timeout=6);_,peer_error=worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error);self.assertEqual(worker.returncode,0,peer_error)

    def test_text_chat_and_sse_match_independent_raw_tokens(self):
        template=self.fixture.path/'chat.jinja'
        template.write_text("{% for m in messages %}{{ m.content }} {% endfor %}{% if enable_thinking %}w3{% endif %}")
        extra=['--chat-template',str(template)]
        service,worker,ready,_=self.start_service(idle='2',extra=extra,worker_extra=extra)
        self.assertTrue(ready['text_chat'])
        def request(route,body):
            return urlopen(Request(ready['url']+route,json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=5)
        body=dict(model='model',messages=[dict(role='user',content='hello world')],max_tokens=3,
                  chat_template_kwargs=dict(enable_thinking=True))
        with request('/v1/chat/completions',body) as response:chat=json.load(response)
        with request('/v1/completions',dict(model='model',prompt=[1,2,3],max_tokens=3)) as response:raw=json.load(response)
        self.assertEqual(chat['choices'][0]['message']['content'],raw['choices'][0]['text'])
        self.assertEqual(chat['usage'],raw['usage'])
        with request('/v1/chat/completions',dict(body,stream=True,stream_options=dict(include_usage=True))) as response:
            wire=response.read().decode()
        self.assertIn('data: [DONE]',wire)
        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        content=''.join(choice.get('delta',{}).get('content','') for event in events for choice in event['choices'])
        self.assertEqual(content,raw['choices'][0]['text']);self.assertEqual(events[-1]['usage'],raw['usage'])
        for messages in ([dict(role='user',content=[dict(type='text',text='hello')])],
                         [dict(role='assistant',content='hello',reasoning_content='world')]):
            with self.assertRaises(HTTPError) as error:request('/v1/chat/completions',dict(body,messages=messages))
            self.assertEqual(error.exception.code,400)
        with request('/v1/completions',dict(model='model',prompt=[1,2,3],max_tokens=3)) as response:
            self.assertEqual(json.load(response)['choices'][0]['text'],raw['choices'][0]['text'])
        _,error=service.communicate(timeout=6);_,peer_error=worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error);self.assertEqual(worker.returncode,0,peer_error)

    def test_admin_drain_stops_both_peers_with_acknowledged_success(self):
        environment=dict(self.fixture.environment,DLLM_API_KEY=secrets.token_urlsafe(32),DLLM_ADMIN_API_KEY=secrets.token_urlsafe(32))
        service,worker,ready,path=self.start_service(idle='0',environment=environment,extra=['--stop-after-drain'])
        payload=json.dumps(dict(model='model',prompt=[1,2],max_tokens=3,stream=True)).encode()
        response=urlopen(Request(ready['url']+'/v1/completions',payload,{'Content-Type':'application/json',
            'Authorization':'Bearer '+environment['DLLM_API_KEY']}),timeout=5)
        with response:
            with urlopen(Request(ready['url']+'/v1/scheduler/drain',b'{}',{'Content-Type':'application/json',
                'Authorization':'Bearer '+environment['DLLM_ADMIN_API_KEY']}),timeout=5) as drained:
                self.assertTrue(json.load(drained)['draining'])
            self.assertIn('data: [DONE]',response.read().decode())
        _,error=service.communicate(timeout=6);peer_output,peer_error=worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error);self.assertEqual(worker.returncode,0,peer_error)
        self.assertEqual(json.loads(path.read_text().splitlines()[-1])['event'],'stopped')
        self.assertEqual(json.loads(peer_output.splitlines()[-1])['event'],'stopped')
        with self.assertRaises(OSError):urlopen(ready['url']+'/health',timeout=1)
        with patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.fixture.args('complete',1234)+['--stop-after-drain'])

    def test_sigterm_closes_service_and_peer(self):
        service,worker,ready,_=self.start_service(idle='0')
        service.terminate();_,error=service.communicate(timeout=6)
        worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error)
        # Interrupted sessions are discarded; the worker observes EOF rather than
        # a fabricated successful STOP acknowledgment.
        self.assertEqual(worker.returncode,2)
        with self.assertRaises(OSError):urlopen(ready['url']+'/health',timeout=1)

    def test_bad_serve_options_and_auth_fail_before_connect(self):
        cases=[['--queue-size','65'],['--queue-timeout','nan'],['--idle-timeout','inf'],
               ['--heartbeat-interval','4'],['--alias',''],['--http-port','65536'],
               ['--prompt','not allowed'],['--prefill-chunk','4097']]
        for extra in cases:
            with self.subTest(extra=extra),patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:cli.main(self.args(1234)+extra)
                self.assertEqual(error.exception.code,2)
        with patch.dict('os.environ',{'DLLM_API_KEY':''}),patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(self.args(1234))
            self.assertEqual(error.exception.code,2)

    def test_decode_turn_flag_reaches_actual_peer_http_runtime(self):
        service,worker,ready,_=self.start_service(extra=['--client-decode-turns','--interleaved-prefill'])
        with urlopen(ready['url']+'/v1/models/status',timeout=3) as response:status=json.load(response)
        self.assertTrue(status['client_decode_turns']);self.assertTrue(status['client_admission_turns'])
        self.assertTrue(status['interleaved_prefill']);self.assertEqual(status['prefill_chunk_size'],1)
        payload=json.dumps(dict(model='model',prompt=[1,2],max_tokens=3)).encode()
        with urlopen(Request(ready['url']+'/v1/completions',payload,{'Content-Type':'application/json'}),timeout=5) as response:result=json.load(response)
        self.assertEqual(result['usage']['completion_tokens'],3)
        _,error=service.communicate(timeout=6);_,peer_error=worker.communicate(timeout=6)
        self.assertEqual(service.returncode,0,error);self.assertEqual(worker.returncode,0,peer_error)
        with patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            for flag in ('--client-decode-turns','--interleaved-prefill'):
                with self.assertRaises(SystemExit):cli.main(self.fixture.args('complete',1234)+[flag])
