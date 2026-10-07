from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import queue
import time
import unittest
from unittest.mock import patch
from dllm.client_prefill_turns import ClientPrefillTurns
from dllm.request_scope import CacheScope,cache_scope
from dllm.scheduler import BatchedRuntime
from dllm.distributed_serving import DistributedRuntime,Request
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from tests import test_scheduler as scheduler_fixtures,test_host_serving as host_fixtures
from tests import test_host_interleaved_prefill as interleaved_fixtures
from tests.test_distributed_serving import Pair
from tests.test_cpu_pipeline_generation import OPTIONS


def pending(scopes,costs):
    return [(i,SimpleNamespace(cache_scope=scope),cost) for i,(scope,cost) in enumerate(zip(scopes,costs))]


class ClientPrefillTurnTests(unittest.TestCase):
    def test_multiple_slots_do_not_multiply_client_service_and_requests_rotate(self):
        a,b=CacheScope(),CacheScope();ready=pending([a,a,a,b],[4]*4);policy=ClientPrefillTurns()
        self.assertEqual([policy.choose(ready)[0] for _ in range(6)],[0,3,1,3,2,3])
        counts={id(a):0,id(b):0}
        for _ in range(100):
            _,request,cost=policy.choose(ready);counts[id(request.cache_scope)]+=cost
            self.assertLessEqual(abs(counts[id(a)]-counts[id(b)]),4)
        self.assertEqual(len(policy.service),2);self.assertEqual(len(policy.recent),4)

    def test_short_chunks_charge_actual_tokens_and_bound_live_history(self):
        a,b=CacheScope(),CacheScope();ready=pending([a,b],[4,1]);policy=ClientPrefillTurns()
        self.assertEqual([policy.choose(ready)[0] for _ in range(6)],[0,1,1,1,1,0])
        for _ in range(50):
            policy.choose(ready)
            self.assertLessEqual(max(value[1] for value in policy.service.values()),4)
        policy.choose(ready[:1]);self.assertEqual(set(policy.service),{id(a)})
        self.assertEqual(len(policy.recent),1)
        self.assertEqual(policy.choose(ready)[0],1) # A rejoining lane starts at the floor.
        self.assertIsNone(policy.choose([]));self.assertFalse(policy.service);self.assertFalse(policy.recent)
        for cost in (0,-1,True,2**20+1):
            with self.assertRaises(ValueError):policy.choose(pending([a],[cost]))

    def test_ordinary_and_paged_chunk_owners_preserve_outputs_and_token_turns(self):
        for executor_type in (scheduler_fixtures.ChunkExecutor,scheduler_fixtures.PagedChunkExecutor):
            with patch('dllm.scheduler.Runtime',scheduler_fixtures.FakeRuntime):
                runtime=BatchedRuntime({'a':'a'},batch_size=4,queue_size=8,prefill_chunk_size=2,client_prefill_turns=True)
            executor=executor_type();runtime.runtime.executor=executor
            if hasattr(executor,'pool'):
                from dllm.pages import PagePool
                executor.pool=PagePool(128,4,executor.context)
            a,b=CacheScope(),CacheScope();counts={id(a):0,id(b):0};observed=[];choose=runtime.prefill_turns.choose
            def select(ready):
                chosen=choose(ready)
                if len({id(r.cache_scope) for _,r,_ in ready})==2:
                    counts[id(chosen[1].cache_scope)]+=chosen[2]
                    self.assertLessEqual(abs(counts[id(a)]-counts[id(b)]),2)
                    observed.append(chosen[0])
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
                    executor.release.set();results=[f.result(6) for f in futures]
                self.assertTrue(observed)
                for events in results:
                    self.assertEqual(''.join(e.get('text','') for e in events),'123')
                    self.assertEqual(events[-1]['usage']['completion_tokens'],3)
                self.assertTrue(runtime.scheduler_status()['client_prefill_turns'])
                if hasattr(executor,'pool'):self.assertEqual(executor.pool.used_tokens,0)
            finally:executor.release.set();runtime.close()
            self.assertFalse(runtime.prefill_turns.service)

    def test_authenticated_cpu_peers_use_client_tokens_and_match_whole_controls(self):
        fixture=host_fixtures.HostServingTests();fixture.batch_size=4
        fixture.runtime_options=dict(interleaved_prefill=True,client_prefill_turns=True)
        fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.pair.chunk_size=2
        a,b=CacheScope(),CacheScope();requests=[Request(tokens,3,time.monotonic(),queue.Queue(9),cache_scope=scope)
            for tokens,scope in (([1]*9,a),([3]*9,a),([2]*7,b))]
        for request in requests:fixture.runtime.requests.put(request)
        choose=fixture.runtime.prefill_turns.choose;counts={id(a):0,id(b):0};trace=[]
        def select(ready):
            chosen=choose(ready)
            if len({id(r.cache_scope) for _,r,_ in ready})==2:
                counts[id(chosen[1].cache_scope)]+=chosen[2]
                self.assertLessEqual(abs(counts[id(a)]-counts[id(b)]),2)
            trace.append((chosen[0].slot,chosen[2]));return chosen
        with patch.object(fixture.runtime.prefill_turns,'choose',side_effect=select):fixture.runtime.run(idle_timeout=.1,heartbeat_interval=.02)
        self.assertEqual([slot for slot,_ in trace[:4]],[0,1,2,1]) # Fair admission places B in slot 1.
        self.assertIn(1,[cost for _,cost in trace])
        source=load_cpu_checkpoint(fixture.fixture.path,quantization='q4_0',max_context=16)
        control=CpuPipelineReferenceGenerator(source,[2**22],vocab_size=fixture.text.vocab_size,**OPTIONS);self.addCleanup(control.cleanup)
        for request in requests:
            events=[]
            while not request.events.empty():events.append(request.events.get_nowait())
            expected=list(control.generate(request.tokens,max_new_tokens=3,chunk_size=2))
            self.assertEqual(events[-1]['usage'],expected[-1]['usage'])
            self.assertEqual(''.join(e.get('text','') for e in events),fixture.text.decode([e['token_id'] for e in expected if 'token_id' in e]))
        self.assertFalse(any(fixture.pair.state.lengths));self.assertFalse(fixture.runtime.prefill_turns.service)

    def test_cancelled_prefill_releases_before_reuse_with_client_policy(self):
        fixture=interleaved_fixtures.HostInterleavedPrefillTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.runtime.client_prefill_turns=True
        fixture.test_partial_cancel_releases_before_reuse_and_never_publishes_chunk_samples()
        self.assertFalse(fixture.runtime.prefill_turns.service)

    def test_bad_or_unsupported_options_reject_before_runtime_or_prefill(self):
        for options in (dict(client_prefill_turns=1,prefill_chunk_size=2),dict(client_prefill_turns=True),
                        dict(client_prefill_turns=True,prefill_chunk_size=2**20+1)):
            with patch('dllm.scheduler.Runtime',side_effect=AssertionError('runtime constructed')):
                with self.assertRaises(ValueError):BatchedRuntime({},**options)
        for options in (dict(client_prefill_turns=True),dict(client_prefill_turns=1,interleaved_prefill=True)):
            with self.assertRaises(ValueError):DistributedRuntime(Pair(),**options)
        with patch('dllm.scheduler.Runtime',scheduler_fixtures.FakeRuntime):
            runtime=BatchedRuntime({'a':'a'},prefill_chunk_size=2,client_prefill_turns=True)
        try:
            with self.assertRaisesRegex(ValueError,'chunk-capable'):list(runtime.generate('a',[{'content':'hi'}],1))
            self.assertFalse(runtime.runtime.executor.entered.is_set())
        finally:runtime.close()

    def test_ordinary_command_forwards_explicit_prefill_policy(self):
        from dllm.cli import main
        with patch('sys.argv',['dllm','serve','--model','a=/local/snapshot','--prefill-chunk-size','2','--client-prefill-turns']), \
                patch('dllm.scheduler.BatchedRuntime') as runtime,patch('dllm.server.serve'):
            main()
        self.assertTrue(runtime.call_args.kwargs['client_prefill_turns'])
        self.assertEqual(runtime.call_args.kwargs['prefill_chunk_size'],2)
