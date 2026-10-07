from concurrent.futures import ThreadPoolExecutor
import json
import queue
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request as HttpRequest,urlopen
from urllib.error import HTTPError

from dllm.cpu_host_worker_build import build_cpu_host_worker
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.cpu_reference_text import CpuReferenceText,read_json
from dllm.distributed_serving import Request
from dllm.host_serving import HostServingPair,HostDistributedRuntime,handler_for_host
from dllm.server import BoundedHTTPServer
from dllm.http_auth import AuthPolicy
from dllm.runtime import BusyError
from tests import test_cpu_host_worker_build as fixtures
from tests.test_cpu_pipeline_generation import tokenizer,OPTIONS


class HostServingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CpuHostWorkerBuildTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        tokenizer(self.fixture.path);self.text=CpuReferenceText(self.fixture.path,read_json(self.fixture.path/'config.json',2**20))
        self.workers=[]
        for transport in self.fixture.transports():
            worker=build_cpu_host_worker(self.fixture.path,0 if transport.rank==0 else 2,2 if transport.rank==0 else 4,
                transport,epoch=1,max_context=16,max_batch=2,quantization='q4_0',mapped_vocab_size=self.text.vocab_size,
                excluded_ids=self.text.excluded_ids,**OPTIONS)
            self.workers.append(worker);self.addCleanup(worker.close)
        self.fixture.together([w.handshake for w in self.workers]);self.peer_errors=[]
        def peer():
            try:
                while not self.workers[1].closed:self.workers[1].step()
            except (RuntimeError,OSError,EOFError,ValueError) as error:self.peer_errors.append(error)
            finally:self.workers[1].close()
        self.peer=threading.Thread(target=peer,daemon=True);self.peer.start()
        self.pair=HostServingPair(self.workers[0],self.text,chunk_size=1)
        self.runtime=HostDistributedRuntime(self.pair,queue_timeout=2,**getattr(self,'runtime_options',{}));self.owner=None;self.errors=[]
        self.addCleanup(self.close)

    def close(self):
        self.runtime.close()
        if self.owner is not None:self.owner.join(3);self.assertFalse(self.owner.is_alive())
        self.peer.join(3);self.assertFalse(self.peer.is_alive())

    def start(self,**options):
        def run():
            try:self.runtime.run(**options)
            except BaseException as error:self.errors.append(error)
        self.owner=threading.Thread(target=run,daemon=True);self.owner.start()

    def serve(self,auth=None):
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for_host(self.runtime,AuthPolicy() if auth is None else auth))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def close():server.shutdown();server.server_close();thread.join(3)
        self.addCleanup(close)
        return f'http://127.0.0.1:{server.server_port}'

    def post(self,base,payload,path='/v1/completions'):
        return urlopen(HttpRequest(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=4)

    def test_real_http_output_streaming_and_truthful_metadata(self):
        self.start();base=self.serve()
        source=load_cpu_checkpoint(self.fixture.path,max_context=16,quantization='q4_0')
        control=CpuPipelineReferenceGenerator(source,[2**22]*2,vocab_size=self.text.vocab_size,**OPTIONS)
        self.addCleanup(control.cleanup)
        expected=list(control.generate([1,2],max_new_tokens=3,chunk_size=1))
        text=self.text.decode([e['token_id'] for e in expected if 'token_id' in e])
        body={'model':'model','prompt':'hello world','max_tokens':3}
        with self.post(base,body) as response:result=json.load(response)
        self.assertEqual(result['choices'][0]['text'],text)
        self.assertEqual(result['usage'],expected[-1]['usage'])
        self.assertEqual(result['dllm_timing']['source_tier'],'cpu_reference')
        self.assertEqual(result['dllm_timing']['kernel_backend'],'cpu-host-reference')
        with self.post(base,dict(body,stream=True,stream_options={'include_usage':True})) as response:wire=response.read().decode()
        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        self.assertEqual(''.join(e['choices'][0].get('text','') for e in events if e['choices']),text)
        self.assertEqual(events[-1]['usage'],expected[-1]['usage']);self.assertIn('data: [DONE]',wire)
        with urlopen(base+'/v1/models/status') as response:status=json.load(response)
        self.assertEqual(status['device'],'cpu');self.assertEqual(status['stage_layers'],[[0,2],[2,4]])
        self.assertFalse(status['official_model_qualified']);self.assertEqual(status['completed'],2)
        self.assertFalse(any(self.pair.state.lengths));self.assertFalse(self.errors)

    def test_http_rejects_unsupported_inputs_before_peer_commands(self):
        self.start();base=self.serve();before=self.pair.state.sequence
        invalid=[{'model':'model','prompt':[60],'max_tokens':1},{'model':'model','prompt':[1],'max_tokens':16},
                 {'model':'model','prompt':[1],'max_tokens':1,'temperature':1},
                 {'model':'unknown','prompt':[1],'max_tokens':1}]
        for body in invalid:
            with self.assertRaises(HTTPError) as error:self.post(base,body)
            self.assertEqual(error.exception.code,400)
        with self.assertRaises(HTTPError) as error:self.post(base,{'model':'model','messages':[{'role':'user','content':'Hi'}],'max_tokens':1},'/v1/chat/completions')
        self.assertEqual(error.exception.code,400);self.assertEqual(self.pair.state.sequence,before)
        with self.assertRaises(HTTPError) as error:urlopen(base+'/v1/models/config')
        self.assertEqual(error.exception.code,400)

    def test_concurrent_http_slots_release_and_reuse(self):
        base=self.serve()
        def request():
            with self.post(base,{'model':'model','prompt':[1,2],'max_tokens':3}) as response:return json.load(response)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(request) for _ in range(2)]
            deadline=time.monotonic()+2
            while self.runtime.requests.qsize()!=2 and time.monotonic()<deadline:time.sleep(.005)
            self.assertEqual(self.runtime.requests.qsize(),2);self.start()
            results=[f.result(timeout=5) for f in futures]
        self.assertEqual([r['dllm_timing']['max_batch_size'] for r in results],[2,2])
        self.assertEqual(results[0]['choices'],results[1]['choices'])
        self.assertFalse(any(self.pair.state.lengths));self.assertEqual(self.runtime.completed,2)
        self.assertEqual(request()['usage']['completion_tokens'],3)

    def test_cancel_before_prefill_never_releases_unowned_slot(self):
        request=Request([1],2,time.monotonic(),queue.Queue(3))
        original=request.started.set
        def cancel():original();request.cancelled.set()
        request.started.set=cancel
        self.runtime.requests.put(request)
        before=self.pair.state.sequence
        with patch.object(self.pair,'release',side_effect=AssertionError('unowned release')):
            self.runtime.run(idle_timeout=.15,heartbeat_interval=1)
        self.assertIsNone(self.runtime.failure);self.assertEqual(self.pair.state.sequence,before)
        self.assertTrue(self.workers[0].ready)

    def test_heartbeat_and_peer_failure_close_epoch(self):
        self.start(heartbeat_interval=.02)
        deadline=time.monotonic()+2
        while not self.runtime.heartbeats and time.monotonic()<deadline:time.sleep(.005)
        self.assertGreater(self.runtime.heartbeats,0)
        self.workers[1].close()
        with self.assertRaises((BusyError,RuntimeError)):list(self.runtime.generate('model',None,2,raw_prompt=[1]))
        self.owner.join(3);self.assertFalse(self.owner.is_alive())
        self.assertTrue(self.workers[0].closed);self.assertTrue(self.runtime.closed.is_set())

    def test_duplicate_adapter_rejected(self):
        with self.assertRaises(ValueError):HostServingPair(self.workers[0],self.text)
        self.assertTrue(self.workers[0].ready)

    def test_single_slot_compatibility_and_health_after_shutdown(self):
        self.runtime.max_batch=1
        self.start();base=self.serve()
        with urlopen(base+'/health') as response:self.assertEqual(json.load(response)['status'],'ok')
        with self.post(base,{'model':'model','prompt':[1,2],'max_tokens':3}) as response:result=json.load(response)
        self.assertEqual(result['usage']['completion_tokens'],3)
        self.assertEqual(result['dllm_timing']['max_batch_size'],1)
        self.assertFalse(any(self.pair.state.lengths))
        self.runtime.close()
        with self.assertRaises(HTTPError) as error:urlopen(base+'/health')
        self.assertEqual(error.exception.code,503)

    def test_real_tls_backpressure_keeps_peer_alive_and_preserves_other_slot(self):
        from dllm.distributed_serving import STREAM_BUFFER_EVENTS
        self.start(heartbeat_interval=.02)
        requests=[]
        def capture(*args,**kwargs):
            request=Request(*args,**kwargs);requests.append(request);return request
        generation=self.runtime.generate('model',None,13,raw_prompt=[1,2])
        self.addCleanup(generation.close)
        with patch('dllm.distributed_serving.Request',side_effect=capture):first=next(generation)
        request=requests[0];deadline=time.monotonic()+3
        while request.events.qsize()<STREAM_BUFFER_EVENTS and time.monotonic()<deadline:time.sleep(.005)
        self.assertEqual(request.events.qsize(),STREAM_BUFFER_EVENTS)
        before=self.pair.state.lengths[0];self.assertLess(before,15)
        survivor=list(self.runtime.generate('model',None,3,raw_prompt=[1,2]))
        self.assertEqual(survivor[-1]['usage']['completion_tokens'],3)
        self.assertEqual(self.pair.state.lengths[0],before)
        deadline=time.monotonic()+3;heartbeats=self.runtime.heartbeats
        while self.runtime.heartbeats==heartbeats and time.monotonic()<deadline:time.sleep(.005)
        self.assertGreater(self.runtime.heartbeats,heartbeats)
        self.assertEqual(self.pair.state.lengths[0],before)
        result=[first,*list(generation)]
        self.assertEqual(result[-1]['usage']['completion_tokens'],13)
        source=load_cpu_checkpoint(self.fixture.path,max_context=16,quantization='q4_0')
        control=CpuPipelineReferenceGenerator(source,[2**22]*2,vocab_size=self.text.vocab_size,**OPTIONS)
        self.addCleanup(control.cleanup)
        expected=list(control.generate([1,2],max_new_tokens=13,chunk_size=1))
        expected_text=self.text.decode([e['token_id'] for e in expected if 'token_id' in e])
        self.assertEqual(''.join(e.get('text','') for e in result),expected_text)
        self.assertFalse(any(self.pair.state.lengths));self.assertFalse(self.errors)

    def test_authenticated_http_clients_get_admission_turns_on_real_peers(self):
        import secrets
        from dllm.http_auth import ClientKey
        a,b=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        base=self.serve(AuthPolicy(clients=(ClientKey('alice',a,8),ClientKey('bob',b,8))))
        order=[];original=self.pair.prefill
        def prefill(tokens,slot):order.append(tokens[0]);return original(tokens,slot)
        def call(key,token):
            body=json.dumps(dict(model='model',prompt=[token],max_tokens=1)).encode()
            with urlopen(HttpRequest(base+'/v1/completions',body,{'Authorization':'Bearer '+key,'Content-Type':'application/json'}),timeout=5) as response:
                return json.load(response)
        with patch.object(self.pair,'prefill',side_effect=prefill),ThreadPoolExecutor(5) as pool:
            futures=[]
            for key,token in ((a,1),(a,3),(a,4),(a,5),(b,2)):
                futures.append(pool.submit(call,key,token));deadline=time.monotonic()+2
                while self.runtime.requests.qsize()<len(futures) and time.monotonic()<deadline:time.sleep(.002)
                self.assertEqual(self.runtime.requests.qsize(),len(futures))
            with self.runtime.requests.mutex:
                scopes=[r.cache_scope for r in self.runtime.requests.queue]
                self.assertTrue(all(s is scopes[0] for s in scopes[:4]));self.assertIsNot(scopes[0],scopes[4])
                self.assertTrue(all(not isinstance(s,str) for s in scopes))
            self.start();results=[f.result(5) for f in futures]
        self.assertEqual(order,[1,2,3,4,5])
        self.assertTrue(all(r['usage']['completion_tokens']==1 for r in results))
        self.assertTrue(self.runtime.manage('status')['client_admission_turns'])
        self.assertFalse(self.runtime.requests.fair.recent);self.assertFalse(any(self.pair.state.lengths));self.assertFalse(self.errors)
