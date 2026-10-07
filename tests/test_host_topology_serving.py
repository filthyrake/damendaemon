from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.cpu_reference_text import CpuReferenceText,read_json
from dllm.host_topology import HostTopology
from dllm.host_serving import HostServingPair,HostServingTopology,HostDistributedRuntime,handler_for_host
from dllm.http_auth import AuthPolicy
from dllm.runtime import BusyError
from dllm.server import BoundedHTTPServer
from tests import test_cpu_host_pipeline_build as builders
from tests.test_cpu_pipeline_generation import tokenizer,OPTIONS
from tests.test_distributed_drain import wait_for


class HostTopologyServingTests(unittest.TestCase):
    def setup_session(self,count=3,*,claim=True):
        fixture=builders.CpuHostPipelineBuildTests();fixture.setUp();self.addCleanup(fixture.doCleanups);tokenizer(fixture.path)
        text=CpuReferenceText(fixture.path,read_json(fixture.path/'config.json',2**20))
        ranges=[(0,1),(1,3),(3,4)] if count==3 else [(i,i+1) for i in range(4)]
        plan=HostTopology(fixture.config,ranges,b't'*32,quantization='q4_0',context=16,slots=2,
            chunk_size=4,frame_bytes=4096,mapped_vocab_size=text.vocab_size,excluded_ids=text.excluded_ids)
        links=fixture.links(plan);workers=[fixture.build(plan,i,group) for i,group in enumerate(links)]
        fixture.together([w.handshake for w in workers])
        self.fixture,self.text,self.workers=fixture,text,workers
        self.peer_errors=[];self.peers=[];self.owner=None;self.errors=[];self.runtime=None
        def peer(worker):
            try:
                while not worker.closed:worker.step()
            except (RuntimeError,ValueError,OSError,EOFError) as error:self.peer_errors.append(error)
            finally:worker.close()
        for worker in workers[1:]:
            thread=threading.Thread(target=peer,args=(worker,),daemon=True);thread.start();self.peers.append(thread)
        def close():
            if self.runtime is not None:self.runtime.close()
            for worker in workers:worker.close()
            for thread in [*self.peers,*([self.owner] if self.owner is not None else [])]:
                thread.join(3);self.assertFalse(thread.is_alive())
        self.addCleanup(close)
        if claim:self.claim()

    def claim(self):
        self.pair=HostServingTopology(self.workers[0],self.text,chunk_size=2)
        self.runtime=HostDistributedRuntime(self.pair,queue_timeout=2,client_decode_turns=True,interleaved_prefill=True)

    def start(self,**options):
        def run():
            try:self.runtime.run(heartbeat_interval=.1,**options)
            except BaseException as error:self.errors.append(error)
        self.owner=threading.Thread(target=run,daemon=True);self.owner.start()

    def serve(self,auth=None):
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for_host(self.runtime,AuthPolicy() if auth is None else auth))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def close():server.shutdown();server.server_close();thread.join(3)
        self.addCleanup(close);return f'http://127.0.0.1:{server.server_port}'

    def post(self,base,body,path='/v1/completions',key=None):
        headers={'Content-Type':'application/json'}
        if key:headers['Authorization']='Bearer '+key
        return urlopen(Request(base+path,json.dumps(body).encode(),headers),timeout=4)

    def test_four_stage_http_json_sse_concurrent_requests_and_metadata(self):
        self.setup_session(4);self.start();base=self.serve()
        source=load_cpu_checkpoint(self.fixture.path,quantization='q4_0',max_context=16)
        control=CpuPipelineReferenceGenerator(source,[2**22],vocab_size=self.text.vocab_size,**OPTIONS)
        self.addCleanup(control.cleanup)
        prompts=[[1,2],[3],[4,5,6,7]];limits=[3,1,4];expected=[]
        for prompt,limit in zip(prompts,limits):
            events=list(control.generate(prompt,max_new_tokens=limit,chunk_size=2))
            expected.append((self.text.decode([e['token_id'] for e in events if 'token_id' in e]),events[-1]['usage']))
        def request(index):
            with self.post(base,dict(model='model',prompt=prompts[index],max_tokens=limits[index])) as response:return json.load(response)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(request,range(3)))
        for result,(text,usage) in zip(results,expected):
            self.assertEqual(result['choices'][0]['text'],text);self.assertEqual(result['usage'],usage)
        with self.post(base,dict(model='model',prompt=prompts[0],max_tokens=3,stream=True,stream_options={'include_usage':True})) as response:wire=response.read().decode()
        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        self.assertEqual(''.join(e['choices'][0].get('text','') for e in events if e['choices']),expected[0][0])
        self.assertEqual(events[-1]['usage'],expected[0][1]);self.assertIn('data: [DONE]',wire)
        with urlopen(base+'/v1/models/status') as response:status=json.load(response)
        self.assertEqual(status['stage_layers'],[[0,1],[1,2],[2,3],[3,4]])
        self.assertEqual(status['topology_identity'],self.workers[0].topology.identity.hex())
        self.assertEqual(status['remote_source_verification'],'not_attested')
        self.assertFalse(status['coordinator_model_identity_verified']);self.assertFalse(status['official_model_qualified'])
        self.assertTrue(all(w.stage.lengths==[0,0] for w in self.workers));self.assertFalse(self.errors)

    def test_drain_auth_release_and_explicit_stop_all_peers(self):
        self.setup_session();inference,admin=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        base=self.serve(AuthPolicy(inference,admin))
        with self.assertRaises(HTTPError) as caught:self.post(base,{},'/v1/scheduler/drain',inference)
        self.assertEqual(caught.exception.code,403);caught.exception.close()
        with ThreadPoolExecutor(3) as pool:
            futures=[pool.submit(lambda:list(self.runtime.generate('model',None,3,raw_prompt=[1,2]))) for _ in range(3)]
            wait_for(lambda:self.runtime.scheduler_status()['queued_generation_requests']==3)
            with self.post(base,{},'/v1/scheduler/drain',admin) as response:self.assertFalse(json.load(response)['drained'])
            self.start(stop_after_drain=True)
            for future in futures:self.assertEqual(future.result(5)[-1]['usage']['completion_tokens'],3)
        self.owner.join(3);self.assertFalse(self.owner.is_alive());self.assertFalse(self.errors)
        self.assertTrue(all(not any(w.stage.lengths) for w in self.workers))
        self.workers[0].stop()
        for peer in self.peers:peer.join(3)
        self.assertFalse(self.peer_errors);self.assertTrue(all(w.closed for w in self.workers))

    def test_cancel_releases_every_stage_and_reuses_session(self):
        self.setup_session();self.start()
        stream=self.runtime.generate('model',None,14,raw_prompt=[1]);next(stream);stream.close()
        wait_for(lambda:not self.runtime.active and not self.runtime.accepted)
        self.assertTrue(all(not any(w.stage.lengths) for w in self.workers))
        self.assertEqual(list(self.runtime.generate('model',None,2,raw_prompt=[2]))[-1]['usage']['completion_tokens'],2)

    def test_middle_worker_loss_fails_health_and_all_links(self):
        self.setup_session();base=self.serve();self.start()
        with patch.object(self.workers[1].stage,'forward_layers',side_effect=RuntimeError('middle worker lost')):
            with self.assertRaises(HTTPError) as failed:self.post(base,dict(model='model',prompt=[1],max_tokens=2))
            failed.exception.close()
        wait_for(lambda:self.workers[0].closed)
        self.assertTrue(all(c.closed for c in self.workers[0].channels))
        with self.assertRaises(HTTPError) as caught:urlopen(base+'/health',timeout=3)
        self.assertEqual(caught.exception.code,503);caught.exception.close()
        with self.assertRaises(BusyError):list(self.runtime.generate('model',None,1,raw_prompt=[1]))

    def test_invalid_adapter_and_request_leave_ready_session_unclaimed_or_unchanged(self):
        self.setup_session(claim=False);worker=self.workers[0]
        for constructor,chunk in ((HostServingPair,2),(HostServingTopology,5)):
            with self.assertRaises(ValueError):constructor(worker,self.text,chunk_size=chunk)
            self.assertIsNone(getattr(worker,'_completion_owner',None))
        self.claim()
        with self.assertRaises(ValueError):HostServingTopology(worker,self.text,chunk_size=2)
        base=self.serve();before=[c.state.sequence for c in worker.channels]
        with self.assertRaises(HTTPError) as caught:self.post(base,dict(model='model',prompt=[60],max_tokens=1))
        self.assertEqual(caught.exception.code,400);caught.exception.close();self.assertEqual(before,[c.state.sequence for c in worker.channels])
        self.assertTrue(worker.ready);self.assertFalse(self.runtime.accepted)
