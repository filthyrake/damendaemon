from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
import secrets
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.client_decode_turns import ClientDecodeTurns,SERVICE_UNITS
from dllm.http_auth import AuthPolicy,ClientKey
from dllm.request_scope import CacheScope,cache_scope
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from tests.test_client_decode_turns import rows
from tests.test_cpu_pipeline_generation import OPTIONS
from tests.test_distributed_drain import wait_for
from tests import test_host_serving as host_fixtures
from tests import test_distributed_decode_turns as distributed_fixtures


class WeightedDecodeTests(unittest.TestCase):
    def test_small_batch_fractional_shares_and_within_client_rotation(self):
        a,b=CacheScope(decode_weight=3),CacheScope();ready=rows([a,b]);policy=ClientDecodeTurns()
        selected=[[slot for slot,_,_ in policy.choose(ready)] for _ in range(6)]
        self.assertEqual(selected,[[0],[0],[0,1]]*2)
        a,b=CacheScope(decode_weight=2),CacheScope();ready=rows([a,a,a,b]);policy=ClientDecodeTurns()
        self.assertEqual([[s for s,_,_ in policy.choose(ready)] for _ in range(3)],[[0,1,3],[0,2,3],[1,2,3]])

    def test_default_equal_shares_rotate_without_tied_timestamp_bias(self):
        a,b=CacheScope(),CacheScope();ready=rows([a,a,a,b,b]);policy=ClientDecodeTurns();counts=[0]*5
        for _ in range(12):
            chosen=policy.choose(ready);self.assertEqual(len(chosen),4)
            for slot,_,_ in chosen:counts[slot]+=1
            self.assertLessEqual(max(counts[:3])-min(counts[:3]),1)
        self.assertEqual(counts,[8,8,8,12,12])

    def test_all_weight_pairs_have_bounded_ratio_lag_and_no_starvation(self):
        for a_weight in range(1,17):
            for b_weight in range(1,17):
                a,b=CacheScope(decode_weight=a_weight),CacheScope(decode_weight=b_weight)
                ready=rows([a,a,b]);policy=ClientDecodeTurns();counts=[0,0]
                for _ in range(96):
                    chosen=policy.choose(ready);self.assertTrue(chosen)
                    for _,request,_ in chosen:counts[0 if request.cache_scope is a else 1]+=1
                    self.assertLess(abs(counts[0]*b_weight-counts[1]*a_weight),max(a_weight,b_weight))
                    for scope,credit in policy.credits.values():self.assertTrue(0<=credit<SERVICE_UNITS//scope.decode_weight)
                self.assertTrue(all(counts));self.assertLessEqual(len(policy.credits),2)

    def test_lane_churn_forgets_fractional_credit_and_single_client_uses_full_batch(self):
        a,b=CacheScope(decode_weight=16),CacheScope();ready=rows([a,b,b]);policy=ClientDecodeTurns()
        for _ in range(15):policy.choose(ready)
        self.assertGreater(policy.credits[id(b)][1],0)
        self.assertEqual(policy.choose(ready[:1]),ready[:1]);self.assertNotIn(id(b),policy.credits)
        self.assertEqual([s for s,_,_ in policy.choose(ready)],[0])
        self.assertEqual(policy.choose(ready[1:]),ready[1:]);self.assertNotIn(id(a),policy.credits)
        policy.choose([]);self.assertFalse(policy.credits);self.assertFalse(policy.recent)

    def test_auth_policy_validation_defaults_immutability_and_scope_capture(self):
        key=secrets.token_urlsafe(32)
        for value in (None,True,0,17,1.5,'3'):
            config=json.dumps({'a':dict(key=key,max_inflight=2,decode_weight=value)})
            with patch.dict('os.environ',{'DLLM_CLIENT_KEYS':config},clear=True):
                with self.assertRaises(ValueError) as caught:AuthPolicy.from_env()
                self.assertNotIn(key,str(caught.exception))
            with self.assertRaises(ValueError):CacheScope(decode_weight=value)
        for value in (1,3,16):
            config=json.dumps({'a':dict(key=key,max_inflight=2,decode_weight=value)})
            with patch.dict('os.environ',{'DLLM_CLIENT_KEYS':config},clear=True):client=AuthPolicy.from_env().clients[0]
            self.assertEqual(client.decode_weight,value);self.assertNotIn(key,repr(client))
        self.assertEqual(ClientKey('a',key,2).decode_weight,1)
        scope=CacheScope(decode_weight=3)
        with self.assertRaises(FrozenInstanceError):scope.decode_weight=1
        from dllm.scheduler import Pending
        from dllm.distributed_serving import Request as DistributedRequest
        import queue,time
        token=cache_scope.set(scope)
        try:pending=Pending('a',[],1);request=DistributedRequest([1],1,time.monotonic(),queue.Queue())
        finally:cache_scope.reset(token)
        self.assertIs(pending.cache_scope,scope);self.assertIs(request.cache_scope,scope)
        self.assertIsNone(cache_scope.get())

    def test_distributed_weights_preserve_output_release_and_disabled_mode(self):
        fixture=distributed_fixtures.DistributedDecodeTurnsTests()
        a,b=CacheScope(decode_weight=3),CacheScope()
        calls=fixture.execute(True,[a,b]);self.assertEqual(calls[:3],[[0],[0],[0,1]])
        self.assertEqual(fixture.execute(False,[a,b]),[[0,1]]*5)

    def test_authenticated_cpu_http_captures_weight_and_preserves_numerical_output(self):
        fixture=host_fixtures.HostServingTests();fixture.runtime_options={'client_decode_turns':True}
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        a,b=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        auth=AuthPolicy(clients=(ClientKey('a',a,2,decode_weight=3),ClientKey('b',b,2)))
        base=fixture.serve(auth);counts={3:0,1:0};observed=[];choose=fixture.runtime.decode_turns.choose
        def select(ready):
            selected=choose(ready)
            if {r.cache_scope.decode_weight for _,r,_ in ready}=={1,3}:
                for _,r,_ in selected:counts[r.cache_scope.decode_weight]+=1
                self.assertLess(abs(counts[3]-3*counts[1]),3);observed.append(len(selected))
            return selected
        fixture.runtime.decode_turns.choose=select
        def request(key,extra=None):
            body=dict(model='model',prompt=[1],max_tokens=12);body.update(extra or {})
            return urlopen(Request(base+'/v1/completions',json.dumps(body).encode(),
                {'Content-Type':'application/json','Authorization':'Bearer '+key}),timeout=5)
        with self.assertRaises(HTTPError) as caught:request(a,dict(decode_weight=16))
        self.assertEqual(caught.exception.code,400);caught.exception.close()
        self.assertFalse(fixture.runtime.accepted)
        def complete(key):
            with request(key) as response:return json.load(response)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(complete,key) for key in (a,b)]
            wait_for(lambda:fixture.runtime.scheduler_status()['queued_generation_requests']==2)
            fixture.start();results=[future.result(6) for future in futures]
        self.assertIn(1,observed);self.assertTrue(counts[3]>counts[1]>0)
        source=load_cpu_checkpoint(fixture.fixture.path,quantization='q4_0',max_context=16)
        control=CpuPipelineReferenceGenerator(source,[2**22],vocab_size=fixture.text.vocab_size,**OPTIONS)
        self.addCleanup(control.cleanup);events=list(control.generate([1],max_new_tokens=12,chunk_size=1))
        expected=fixture.text.decode([e['token_id'] for e in events if 'token_id' in e])
        for result in results:
            self.assertEqual(result['choices'][0]['text'],expected);self.assertEqual(result['usage'],events[-1]['usage'])
            self.assertNotIn(a,json.dumps(result));self.assertNotIn(b,json.dumps(result))
        wait_for(lambda:not fixture.runtime.active)
        self.assertFalse(any(fixture.pair.state.lengths));self.assertFalse(fixture.errors)
