from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import queue
import unittest
from unittest.mock import patch

from dllm.client_decode_turns import ClientDecodeTurns
from dllm.scheduler import BatchedRuntime,Pending
from dllm.request_scope import CacheScope,cache_scope
from tests import test_scheduler as fixtures


def rows(scopes):
    return [(i,SimpleNamespace(cache_scope=scope),1) for i,scope in enumerate(scopes)]


class ClientDecodeTurnTests(unittest.TestCase):
    def test_unequal_clients_get_equal_decode_rows_and_rotate_requests(self):
        a,b=object(),object();ready=rows([a,a,a,b]);policy=ClientDecodeTurns()
        rounds=[[slot for slot,_,_ in policy.choose(ready)] for _ in range(6)]
        self.assertEqual(rounds,[[0,3],[1,3],[2,3]]*2)
        self.assertEqual(len(policy.recent),4)

    def test_largest_equal_share_and_single_lane_full_batch(self):
        a,b,c=object(),object(),object();ready=rows([a,a,a,a,b,b,c,c,c]);policy=ClientDecodeTurns()
        selected=policy.choose(ready)
        self.assertEqual([slot for slot,_,_ in selected],[0,1,4,5,6,7])
        selected=policy.choose(ready)
        self.assertEqual([slot for slot,_,_ in selected],[2,3,4,5,6,8])
        self.assertEqual(policy.choose(ready[:4]),ready[:4])
        self.assertEqual(len(policy.recent),4)
        self.assertEqual(policy.choose([]),[]);self.assertFalse(policy.recent)

    def test_identity_not_equality_and_no_debt_for_absent_lane(self):
        a,b=[],[];ready=rows([a,a,b]);policy=ClientDecodeTurns()
        self.assertEqual([s for s,_,_ in policy.choose(ready)],[0,2])
        self.assertEqual(policy.choose(ready[:2]),ready[:2])
        self.assertEqual(len(policy.recent),2)
        self.assertEqual([s for s,_,_ in policy.choose(ready)],[0,2])
        policy.reset();self.assertFalse(policy.recent)
        anonymous=rows([None]*4);self.assertEqual(policy.choose(anonymous),anonymous)

    def test_backpressured_requests_do_not_block_other_lanes(self):
        runtime=BatchedRuntime.__new__(BatchedRuntime)
        runtime.client_decode_turns=True;runtime.decode_turns=ClientDecodeTurns()
        a,b=CacheScope(),CacheScope()
        requests=[Pending('a',[],5,cache_scope=s,events=queue.Queue(maxsize=1)) for s in (a,a,b)]
        active={i:(r,1) for i,r in enumerate(requests)}
        requests[2].events.put({})
        self.assertEqual([s for s,_,_ in runtime._ready_decode(active)],[0,1])
        requests[2].events.get()
        self.assertEqual([s for s,_,_ in runtime._ready_decode(active)],[0,2])
        runtime.client_decode_turns=False
        self.assertEqual([s for s,_,_ in runtime._ready_decode(active)],[0,1,2])

    def test_real_scheduler_paths_preserve_outputs_and_equal_ready_shares(self):
        for executor_type in (fixtures.FakeExecutor,fixtures.ChunkExecutor,fixtures.FakePagedExecutor,fixtures.PagedChunkExecutor):
            with self.subTest(path=executor_type.__name__):
                with patch('dllm.scheduler.Runtime',fixtures.FakeRuntime):
                    runtime=BatchedRuntime({'a':'a'},batch_size=4,queue_size=8,client_decode_turns=True,
                        prefill_chunk_size=1 if issubclass(executor_type,fixtures.ChunkExecutor) else 0)
                executor=executor_type();runtime.runtime.executor=executor
                if hasattr(executor,'pool'):
                    from dllm.pages import PagePool
                    executor.pool=PagePool(128,4,executor.context)
                a,b=CacheScope(),CacheScope();observed=[];choose=runtime.decode_turns.choose
                def select(ready):
                    selected=choose(ready)
                    counts={id(r.cache_scope):0 for _,r,_ in ready}
                    for _,r,_ in selected:counts[id(r.cache_scope)]+=1
                    if len(counts)>1:
                        self.assertEqual(len(set(counts.values())),1)
                        observed.append((len(ready),len(selected)))
                    return selected
                runtime.decode_turns.choose=select
                def run(scope):
                    token=cache_scope.set(scope)
                    try:return list(runtime.generate('a',[{'content':'hi'}],7))
                    finally:cache_scope.reset(token)
                try:
                    with ThreadPoolExecutor(4) as pool:
                        futures=[pool.submit(run,a)]
                        self.assertTrue(executor.entered.wait(2))
                        for scope in (a,a,b):futures.append(pool.submit(run,scope))
                        with runtime.condition:
                            self.assertTrue(runtime.condition.wait_for(lambda:len(runtime.pending)==3,timeout=2))
                        executor.release.set()
                        results=[f.result(timeout=5) for f in futures]
                    self.assertTrue(any(before>after for before,after in observed),observed)
                    for events in results:
                        self.assertEqual(events[-1]['usage']['completion_tokens'],7)
                        self.assertEqual(''.join(e.get('text','') for e in events),'1234567')
                    self.assertTrue(runtime.scheduler_status()['client_decode_turns'])
                    if hasattr(executor,'pool'):self.assertEqual(executor.pool.used_tokens,0)
                finally:executor.release.set();runtime.close()
                self.assertFalse(runtime.decode_turns.recent)

    def test_option_validation_and_cli_forwarding(self):
        for invalid in (1,None,'yes'):
            with patch('dllm.scheduler.Runtime',side_effect=AssertionError('runtime created')):
                with self.assertRaises(ValueError):BatchedRuntime({},client_decode_turns=invalid)
        from dllm.cli import main
        for flag in ([],['--client-decode-turns']):
            with patch('sys.argv',['dllm','serve','--model','a=/local/snapshot',*flag]),patch('dllm.scheduler.BatchedRuntime') as runtime,patch('dllm.server.serve'):
                main()
            self.assertEqual(runtime.call_args.kwargs['client_decode_turns'],bool(flag))
