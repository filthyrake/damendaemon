from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
import secrets
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from damendaemon.client_prefill_turns import ClientPrefillTurns,SERVICE_UNITS
from damendaemon.http_auth import AuthPolicy,ClientKey
from damendaemon.request_scope import CacheScope,cache_scope
from damendaemon.scheduler import BatchedRuntime
from damendaemon.cpu_hybrid_checkpoint import load_cpu_checkpoint
from damendaemon.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from tests.test_client_prefill_turns import pending
from tests.test_cpu_pipeline_generation import OPTIONS
from tests.test_distributed_drain import wait_for
from tests import test_scheduler as scheduler_fixtures,test_host_serving as host_fixtures


class WeightedPrefillTests(unittest.TestCase):
    def test_all_weight_pairs_and_unequal_chunks_bound_normalized_service(self):
        for a_weight in range(1,17):
            for b_weight in range(1,17):
                a,b=CacheScope(prefill_weight=a_weight),CacheScope(prefill_weight=b_weight)
                for sizes in ((2,2),(7,1),(3,5)):
                    ready=pending([a,b],sizes);policy=ClientPrefillTurns();counts=[0,0]
                    for _ in range(192):
                        slot,request,cost=policy.choose(ready);counts[slot]+=cost
                        self.assertLessEqual(abs(counts[0]*b_weight-counts[1]*a_weight),max(sizes[0]*b_weight,sizes[1]*a_weight))
                        self.assertLessEqual(max(value[1] for value in policy.service.values()),max(sizes)*SERVICE_UNITS)
                    self.assertTrue(all(counts));self.assertEqual(len(policy.service),2)

    def test_client_rotation_anonymous_defaults_and_independent_decode_weight(self):
        a,b=CacheScope(prefill_weight=3,decode_weight=1),CacheScope(prefill_weight=1,decode_weight=16)
        ready=pending([a,a,a,b],[2]*4);policy=ClientPrefillTurns()
        self.assertEqual([policy.choose(ready)[0] for _ in range(8)],[0,3,1,2,3,0,1,2])
        ready=pending([None,None,CacheScope(decode_weight=16)],[2]*3);policy=ClientPrefillTurns()
        self.assertEqual([policy.choose(ready)[0] for _ in range(6)],[0,2,1,2,0,2])

    def test_churn_forgets_history_and_invalid_weights_do_not_mutate_policy(self):
        a,b=CacheScope(prefill_weight=16),CacheScope();ready=pending([a,b],[1,4]);policy=ClientPrefillTurns()
        for _ in range(12):policy.choose(ready)
        policy.choose(ready[:1]);self.assertEqual(set(policy.service),{id(a)})
        self.assertEqual(policy.choose(ready)[0],1)
        service=dict(policy.service);recent=dict(policy.recent);turn=policy.turn
        for value in (None,True,0,17,1.5,'2'):
            forged=SimpleNamespace(prefill_weight=value)
            with self.assertRaises(ValueError):policy.choose(pending([forged],[1]))
            self.assertEqual(policy.service,service);self.assertEqual(policy.recent,recent);self.assertEqual(policy.turn,turn)
        self.assertIsNone(policy.choose([]));self.assertFalse(policy.service);self.assertFalse(policy.recent)
        class EqualScope:
            prefill_weight=2
            def __eq__(self,other):return True
        scopes=[EqualScope(),EqualScope()];policy.choose(pending(scopes,[1,1]));self.assertEqual(len(policy.service),2)

    def test_auth_defaults_validation_immutability_and_scope_capture(self):
        key=secrets.token_urlsafe(32)
        for value in (None,True,0,17,1.5,'2'):
            with patch.dict('os.environ',{'DAMENDAEMON_CLIENT_KEYS':json.dumps({'a':dict(key=key,max_inflight=2,prefill_weight=value)})},clear=True):
                with self.assertRaises(ValueError) as caught:AuthPolicy.from_env()
            self.assertNotIn(key,str(caught.exception))
            with self.assertRaises(ValueError):CacheScope(prefill_weight=value)
            with self.assertRaises(ValueError):ClientKey('a',key,2,prefill_weight=value)
        for value in (1,3,16):
            with patch.dict('os.environ',{'DAMENDAEMON_CLIENT_KEYS':json.dumps({'a':dict(key=key,max_inflight=2,prefill_weight=value,decode_weight=2)})},clear=True):
                client=AuthPolicy.from_env().clients[0]
            self.assertEqual((client.prefill_weight,client.decode_weight),(value,2));self.assertNotIn(key,repr(client))
        self.assertEqual(ClientKey('a',key,2).prefill_weight,1)
        scope=CacheScope(prefill_weight=3)
        with self.assertRaises(FrozenInstanceError):scope.prefill_weight=1
        from damendaemon.scheduler import Pending
        from damendaemon.distributed_serving import Request as DistributedRequest
        import queue,time
        token=cache_scope.set(scope)
        try:first=Pending('a',[],1);second=DistributedRequest([1],1,time.monotonic(),queue.Queue())
        finally:cache_scope.reset(token)
        self.assertIs(first.cache_scope,scope);self.assertIs(second.cache_scope,scope);self.assertIsNone(cache_scope.get())

    def test_ordinary_and_paged_weighted_prefill_preserve_output_and_release(self):
        for executor_type in (scheduler_fixtures.ChunkExecutor,scheduler_fixtures.PagedChunkExecutor):
            with patch('damendaemon.scheduler.Runtime',scheduler_fixtures.FakeRuntime):
                runtime=BatchedRuntime({'a':'a'},batch_size=4,queue_size=8,prefill_chunk_size=2,client_prefill_turns=True)
            executor=executor_type();runtime.runtime.executor=executor
            if hasattr(executor,'pool'):
                from damendaemon.pages import PagePool
                executor.pool=PagePool(128,4,executor.context)
            a,b=CacheScope(prefill_weight=3),CacheScope();counts={id(a):0,id(b):0};observed=[];choose=runtime.prefill_turns.choose
            def select(ready):
                chosen=choose(ready)
                if len({id(r.cache_scope) for _,r,_ in ready})==2:
                    counts[id(chosen[1].cache_scope)]+=chosen[2]
                    self.assertLessEqual(abs(counts[id(a)]-3*counts[id(b)]),6);observed.append(chosen[0])
                else:counts.update({id(a):0,id(b):0})
                return chosen
            runtime.prefill_turns.choose=select
            def run(scope,prompt):
                token=cache_scope.set(scope)
                try:return list(runtime.generate('a',[{'content':prompt}],3))
                finally:cache_scope.reset(token)
            try:
                with ThreadPoolExecutor(4) as pool:
                    futures=[pool.submit(run,a,'x'*13)];self.assertTrue(executor.entered.wait(2))
                    for scope,prompt in ((a,'x'*13),(a,'x'*13),(b,'y'*11)):futures.append(pool.submit(run,scope,prompt))
                    with runtime.condition:self.assertTrue(runtime.condition.wait_for(lambda:len(runtime.pending)==3,timeout=2))
                    executor.release.set();results=[future.result(6) for future in futures]
                self.assertTrue(observed)
                for events in results:self.assertEqual(''.join(event.get('text','') for event in events),'123')
                if hasattr(executor,'pool'):self.assertEqual(executor.pool.used_tokens,0)
            finally:executor.release.set();runtime.close()
            self.assertFalse(runtime.prefill_turns.service)

    def test_authenticated_cpu_http_weight_capture_and_numerical_controls(self):
        fixture=host_fixtures.HostServingTests();fixture.batch_size=4
        fixture.runtime_options=dict(interleaved_prefill=True,client_prefill_turns=True)
        fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.pair.chunk_size=2
        a,b=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        auth=AuthPolicy(clients=(ClientKey('a',a,2,prefill_weight=3),ClientKey('b',b,2,prefill_weight=1,decode_weight=16)))
        base=fixture.serve(auth);counts={3:0,1:0};observed=[];choose=fixture.runtime.prefill_turns.choose
        def select(ready):
            chosen=choose(ready)
            if {r.cache_scope.prefill_weight for _,r,_ in ready}=={1,3}:
                counts[chosen[1].cache_scope.prefill_weight]+=chosen[2]
                self.assertLessEqual(abs(counts[3]-3*counts[1]),6);observed.append(chosen[2])
            else:counts.update({3:0,1:0})
            return chosen
        fixture.runtime.prefill_turns.choose=select
        def request(key,extra=None):
            body=dict(model='model',prompt=[1]*13,max_tokens=3);body.update(extra or {})
            return urlopen(Request(base+'/v1/completions',json.dumps(body).encode(),
                {'Content-Type':'application/json','Authorization':'Bearer '+key}),timeout=6)
        for extra in (dict(prefill_weight=16),dict(cache_scope='a')):
            with self.assertRaises(HTTPError) as error:request(a,extra)
            self.assertEqual(error.exception.code,400);error.exception.close()
        self.assertFalse(fixture.runtime.accepted)
        def complete(key):
            with request(key) as response:return json.load(response)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(complete,key) for key in (a,b)]
            wait_for(lambda:fixture.runtime.scheduler_status()['queued_generation_requests']==2)
            fixture.start();results=[future.result(6) for future in futures]
        self.assertTrue(observed)
        source=load_cpu_checkpoint(fixture.fixture.path,quantization='q4_0',max_context=16)
        control=CpuPipelineReferenceGenerator(source,[2**22],vocab_size=fixture.text.vocab_size,**OPTIONS);self.addCleanup(control.cleanup)
        events=list(control.generate([1]*13,max_new_tokens=3,chunk_size=2))
        expected=fixture.text.decode([event['token_id'] for event in events if 'token_id' in event])
        for result in results:
            self.assertEqual(result['choices'][0]['text'],expected);self.assertEqual(result['usage'],events[-1]['usage'])
            self.assertNotIn(a,json.dumps(result));self.assertNotIn(b,json.dumps(result))
        wait_for(lambda:not fixture.runtime.active);self.assertFalse(any(fixture.pair.state.lengths));self.assertFalse(fixture.errors)

    def test_disabled_scheduler_does_not_use_weight_policy(self):
        with patch('damendaemon.scheduler.Runtime',scheduler_fixtures.FakeRuntime):
            runtime=BatchedRuntime({'a':'a'},prefill_chunk_size=2)
        executor=scheduler_fixtures.ChunkExecutor();executor.release.set();runtime.runtime.executor=executor
        token=cache_scope.set(CacheScope(prefill_weight=16))
        try:
            with patch.object(runtime.prefill_turns,'choose',side_effect=AssertionError('disabled policy')):
                events=list(runtime.generate('a',[{'content':'abc'}],3))
            self.assertEqual(''.join(event.get('text','') for event in events),'123')
        finally:cache_scope.reset(token);runtime.close()

    def test_weighted_partial_cancellation_releases_before_reuse(self):
        from tests.test_host_interleaved_prefill import HostInterleavedPrefillTests
        fixture=HostInterleavedPrefillTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.runtime.client_prefill_turns=True;original=fixture.request
        scopes={1:CacheScope(prefill_weight=16),2:CacheScope(),3:CacheScope(prefill_weight=3)}
        def request(tokens,maximum):
            scope=cache_scope.set(scopes[tokens[0]])
            try:return original(tokens,maximum)
            finally:cache_scope.reset(scope)
        with patch.object(fixture,'request',request):
            fixture.test_partial_cancel_releases_before_reuse_and_never_publishes_chunk_samples()
        self.assertFalse(fixture.runtime.prefill_turns.service);self.assertFalse(fixture.runtime.prefill_turns.recent)
