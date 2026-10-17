from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from types import SimpleNamespace
import threading
import time
import unittest

from damendaemon.distributed_serving import DistributedRuntime
from damendaemon.runtime import BusyError


class Model:
    max_context=32
    quantization='q8_0'
    eos={99}
    tokenizer=SimpleNamespace(decode=lambda tokens,**kwargs:''.join(chr(t) for t in tokens if t!=99))
    def raw_tokens(self,prompt):
        tokens=list(prompt) if isinstance(prompt,list) else [ord(c) for c in prompt]
        if any(type(t) is not int or not 0<=t<128 for t in tokens):raise ValueError('invalid token')
        return tokens
    def chat_tokens(self,messages,**kwargs):return [7,8]


class Pair:
    rank=0
    identity=b'x'*32
    def __init__(self):
        self.model=Model();self.state=SimpleNamespace(failed=False,lengths=[0])
        self.calls=[];self.output=0;self.releases=0;self.fail=False
        self.release_entered=threading.Event();self.release_gate=None
        self.pause_step=None;self.step_entered=threading.Event();self.step_gate=threading.Event()
    def step(self,tokens):
        self.calls.append(('step',threading.get_ident(),list(tokens)))
        if self.fail:self.state.failed=True;raise RuntimeError('injected transport failure')
        if len([c for c in self.calls if c[0]=='step'])==self.pause_step:
            self.step_entered.set();assert self.step_gate.wait(2)
        if self.state.lengths[0]==0:self.output=0;self.state.lengths[0]=len(tokens)
        else:self.state.lengths[0]+=1
        self.output+=1;return 64+self.output
    def release(self):
        self.release_entered.set()
        if self.release_gate is not None:assert self.release_gate.wait(2)
        self.calls.append(('release',threading.get_ident(),None))
        self.releases+=1;self.state.lengths[0]=0
    def heartbeat(self):self.calls.append(('heartbeat',threading.get_ident(),None))


class RuntimeHelpers:
    def start(self,runtime,**options):
        errors=[]
        def run():
            try:runtime.run(**options)
            except BaseException as error:errors.append(error)
        owner=threading.Thread(target=run);owner.start()
        def cleanup():
            runtime.close();owner.join(3);self.assertFalse(owner.is_alive())
        self.addCleanup(cleanup)
        return owner,errors
    def queued(self,runtime,count=1):
        deadline=time.monotonic()+2
        while runtime.requests.qsize()<count and time.monotonic()<deadline:time.sleep(.001)
        self.assertEqual(runtime.requests.qsize(),count)



class DistributedServingTests(RuntimeHelpers, unittest.TestCase):
    def test_final_waits_for_peer_release_and_gpu_calls_have_one_owner(self):
        pair=Pair();pair.release_gate=threading.Event();runtime=DistributedRuntime(pair)
        owner,errors=self.start(runtime)
        generation=runtime.generate('model',None,2,raw_prompt=[1])
        self.assertEqual(next(generation)['text'],'A');self.assertEqual(next(generation)['text'],'B')
        self.assertTrue(pair.release_entered.wait(2))
        with ThreadPoolExecutor(1) as pool:
            final=pool.submit(next,generation)
            self.assertFalse(final.done());pair.release_gate.set()
            result=final.result(2)
        self.assertEqual(result['usage']['completion_tokens'],2)
        self.assertEqual(pair.releases,1);self.assertEqual(runtime.completed,1)
        generation.close()
        self.assertEqual({call[1] for call in pair.calls},{owner.ident});self.assertEqual(errors,[])

    def test_disconnect_releases_before_next_request(self):
        pair=Pair();pair.pause_step=2;runtime=DistributedRuntime(pair)
        _,errors=self.start(runtime)
        generation=runtime.generate('model',None,20,raw_prompt=[1])
        self.assertEqual(next(generation)['text'],'A')
        self.assertTrue(pair.step_entered.wait(2));generation.close();pair.step_gate.set()
        result=list(runtime.generate('model',None,2,raw_prompt=[2,3]))
        self.assertEqual(''.join(e.get('text','') for e in result),'AB')
        self.assertEqual(pair.releases,2);self.assertEqual(runtime.completed,1);self.assertEqual(errors,[])
        calls=[c[0] for c in pair.calls]
        self.assertEqual(calls[:4],['step','step','release','step'])

    def test_queue_bound_and_close_wake_waiters(self):
        pair=Pair();runtime=DistributedRuntime(pair,queue_size=1)
        with ThreadPoolExecutor(1) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1])))
            self.queued(runtime)
            with self.assertRaises(BusyError):next(runtime.generate('model',None,2,raw_prompt=[1]))
            runtime.close()
            with self.assertRaises(BusyError):first.result(2)
        self.assertEqual(pair.calls,[])

    def test_queue_deadline_ends_at_admission_not_first_token(self):
        pair=Pair();pair.pause_step=1;runtime=DistributedRuntime(pair,queue_timeout=.05)
        self.start(runtime)
        with ThreadPoolExecutor(1) as pool:
            result=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1])))
            self.assertTrue(pair.step_entered.wait(2))
            with self.assertRaises(FutureTimeout):result.result(.15)
            pair.step_gate.set()
            self.assertEqual(result.result(2)[-1]['usage']['completion_tokens'],2)

    def test_expired_waiter_releases_queue_capacity_before_active_step_finishes(self):
        pair=Pair();pair.pause_step=1;runtime=DistributedRuntime(pair,queue_size=1,queue_timeout=.1)
        self.start(runtime)
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1])))
            try:
                self.assertTrue(pair.step_entered.wait(2))
                expired=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[2])))
                with self.assertRaisesRegex(BusyError,'deadline'):expired.result(2)
                self.assertEqual(runtime.requests.qsize(),0)
                self.assertEqual(len(runtime.accepted),1)
                replacement=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[3])))
                self.queued(runtime)
                self.assertFalse(first.done())
                pair.step_gate.set()
                first.result(2);replacement.result(2)
            finally:pair.step_gate.set()
        self.assertEqual(pair.releases,2)

    def test_expired_queue_never_touches_pair(self):
        pair=Pair();runtime=DistributedRuntime(pair,queue_timeout=1)
        with ThreadPoolExecutor(1) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1])))
            self.queued(runtime)
            request=runtime.requests.get_nowait();request.submitted-=2;runtime.requests.put_nowait(request)
            self.start(runtime)
            with self.assertRaises(BusyError):first.result(2)
        self.assertEqual(pair.calls,[])

    def test_transport_failure_stops_and_fails_queued_work(self):
        pair=Pair();pair.fail=True;runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1])))
            self.queued(runtime)
            second=pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[2])))
            self.queued(runtime,2);owner,errors=self.start(runtime)
            with self.assertRaises(RuntimeError):first.result(2)
            with self.assertRaises(BusyError):second.result(2)
        owner.join(2)
        self.assertTrue(runtime.closed.is_set());self.assertIsNotNone(runtime.failure)
        self.assertEqual(len(errors),1);self.assertEqual([c[0] for c in pair.calls],['step'])

    def test_unsupported_options_and_bad_prompts_reject_before_admission(self):
        runtime=DistributedRuntime(Pair())
        for options in ({'temperature':.5},{'seed':42},{'tools':[]},{'response_format':{}},
                        {'logprobs':True},{'stop':['x']},{'raw_prompt':[128]}, {'raw_prompt':[1]*32}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                next(runtime.generate('model',None,2,**({'raw_prompt':[1]}|options)))
        self.assertEqual(runtime.requests.qsize(),0)
        with self.assertRaises(ValueError):runtime.manage('unload',model='model')

    def test_idle_heartbeats_and_bounded_idle_shutdown(self):
        pair=Pair();runtime=DistributedRuntime(pair)
        runtime.run(idle_timeout=.35,heartbeat_interval=.1)
        self.assertTrue(runtime.closed.is_set())
        self.assertGreaterEqual(runtime.heartbeats,1)
        self.assertTrue(all(c[0]=='heartbeat' for c in pair.calls))


class BatchedPair(Pair):
    def __init__(self):
        super().__init__()
        self.max_batch=4
        self.state.lengths=[0]*4
        self.outputs=[0]*4
        self.decode_gate=None
        self.decode_entered=threading.Event()
    def prefill(self,tokens,slot):
        assert self.state.lengths[slot]==0
        self.calls.append(('prefill',threading.get_ident(),slot))
        self.state.lengths[slot]=len(tokens);self.outputs[slot]=65
        return self.outputs[slot]
    def decode(self,tokens,slots):
        self.calls.append(('decode',threading.get_ident(),list(slots)))
        self.decode_entered.set()
        if self.decode_gate is not None:assert self.decode_gate.wait(2)
        if self.fail:self.state.failed=True;raise RuntimeError('injected batch failure')
        for token,slot in zip(tokens,slots):
            assert token==self.outputs[slot] and self.state.lengths[slot]>0
            self.state.lengths[slot]+=1;self.outputs[slot]+=1
        return [self.outputs[slot] for slot in slots]
    def release(self,slots):
        self.calls.append(('release',threading.get_ident(),list(slots)))
        self.release_entered.set()
        if self.release_gate is not None:assert self.release_gate.wait(2)
        for slot in slots:self.state.lengths[slot]=0
        self.releases+=1


class DistributedBatchTests(RuntimeHelpers, unittest.TestCase):
    def test_batch_shrinks_and_all_final_results_follow_release(self):
        pair=BatchedPair();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda n=n:list(runtime.generate('model',None,n,raw_prompt=[1])))
                     for n in (2,5,3,4)]
            self.queued(runtime,4);owner,errors=self.start(runtime)
            results=[future.result(2) for future in futures]
        self.assertEqual([''.join(e.get('text','') for e in r) for r in results],['AB','ABCDE','ABC','ABCD'])
        self.assertEqual([r[-1]['timing']['max_batch_size'] for r in results],[4]*4)
        self.assertEqual([c[2] for c in pair.calls if c[0]=='decode'],[[0,1,2,3],[1,2,3],[1,3],[1]])
        self.assertEqual(pair.state.lengths,[0]*4)
        self.assertEqual({c[1] for c in pair.calls},{owner.ident});self.assertEqual(errors,[])

    def test_cancel_one_slot_during_decode_does_not_cancel_its_peers(self):
        pair=BatchedPair();pair.decode_gate=threading.Event();runtime=DistributedRuntime(pair)
        generation=runtime.generate('model',None,20,raw_prompt=[1])
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(next,generation);self.queued(runtime)
            survivor=pool.submit(lambda:list(runtime.generate('model',None,4,raw_prompt=[2])))
            self.queued(runtime,2);_,errors=self.start(runtime)
            self.assertEqual(first.result(2)['text'],'A')
            self.assertTrue(pair.decode_entered.wait(2));generation.close();pair.decode_gate.set()
            result=survivor.result(2)
        self.assertEqual(''.join(e.get('text','') for e in result),'ABCD')
        self.assertEqual(pair.state.lengths,[0]*4);self.assertEqual(runtime.completed,1)
        self.assertEqual(errors,[])

    def test_batch_transport_failure_fails_every_active_request(self):
        pair=BatchedPair();pair.fail=True;runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,4,raw_prompt=[1]))) for _ in range(4)]
            self.queued(runtime,4);owner,errors=self.start(runtime)
            for future in futures:
                with self.assertRaises(RuntimeError):future.result(2)
        owner.join(2)
        self.assertTrue(runtime.closed.is_set());self.assertEqual(len(errors),1)
        self.assertFalse(any(c[0]=='release' for c in pair.calls))

    def test_retired_slot_is_reused_while_long_requests_keep_decoding(self):
        pair=BatchedPair();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(5) as pool:
            futures=[pool.submit(lambda n=n:list(runtime.generate('model',None,n,raw_prompt=[1])))
                     for n in (2,5,5,5,3)]
            self.queued(runtime,5);_,errors=self.start(runtime)
            results=[future.result(2) for future in futures]
        self.assertEqual([''.join(e.get('text','') for e in r) for r in results],['AB','ABCDE','ABCDE','ABCDE','ABC'])
        calls=[(c[0],c[2]) for c in pair.calls]
        released=calls.index(('release',[0]))
        self.assertEqual(calls[released+1],('prefill',0))
        self.assertEqual(calls[released+2],('decode',[1,2,3,0]))
        self.assertEqual(pair.state.lengths,[0]*4);self.assertEqual(errors,[])

    def test_batch_finals_are_not_visible_until_release_acknowledges(self):
        pair=BatchedPair();pair.release_gate=threading.Event();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,2,raw_prompt=[1]))) for _ in range(4)]
            self.queued(runtime,4);self.start(runtime)
            self.assertTrue(pair.release_entered.wait(2))
            self.assertTrue(all(not future.done() for future in futures))
            pair.release_gate.set()
            self.assertTrue(all(future.result(2)[-1]['finish_reason']=='length' for future in futures))

    def test_one_token_requests_release_without_empty_decode(self):
        pair=BatchedPair();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,1,raw_prompt=[1]))) for _ in range(4)]
            self.queued(runtime,4);self.start(runtime)
            results=[future.result(2) for future in futures]
        self.assertEqual([r[-1]['usage']['completion_tokens'] for r in results],[1]*4)
        self.assertFalse(any(c[0]=='decode' for c in pair.calls))
        self.assertEqual(pair.state.lengths,[0]*4)

    def test_close_during_batch_releases_all_slots_without_success(self):
        pair=BatchedPair();pair.decode_gate=threading.Event();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,8,raw_prompt=[1]))) for _ in range(4)]
            self.queued(runtime,4);owner,errors=self.start(runtime)
            self.assertTrue(pair.decode_entered.wait(2));runtime.close();pair.decode_gate.set()
            for future in futures:
                with self.assertRaises(BusyError):future.result(2)
        owner.join(2)
        self.assertEqual(pair.state.lengths,[0]*4);self.assertEqual(runtime.completed,0)
        self.assertEqual(errors,[])


class PackedPair(BatchedPair):
    packed_prefill=True
    def prefill_batch(self,prompts,slots):
        assert len(prompts)==len(slots) and sum(map(len,prompts))<=self.model.max_context
        assert all(self.state.lengths[slot]==0 for slot in slots)
        self.calls.append(('prefill_batch',threading.get_ident(),list(slots)))
        for prompt,slot in zip(prompts,slots):
            self.state.lengths[slot]=len(prompt);self.outputs[slot]=65
        return [65]*len(slots)


class DistributedPackedTests(RuntimeHelpers, unittest.TestCase):
    def test_packed_prompts_preserve_results_and_report_actual_group_size(self):
        pair=PackedPair();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,3,raw_prompt=[1,2]))) for _ in range(4)]
            self.queued(runtime,4);owner,errors=self.start(runtime)
            results=[future.result(2) for future in futures]
        self.assertEqual([r[-1]['timing']['prefill_batch_size'] for r in results],[4]*4)
        self.assertEqual([''.join(e.get('text','') for e in r) for r in results],['ABC']*4)
        self.assertEqual([c[2] for c in pair.calls if c[0]=='prefill_batch'],[[0,1,2,3]])
        self.assertEqual({c[1] for c in pair.calls},{owner.ident});self.assertEqual(errors,[])

    def test_group_budget_splits_without_reordering_requests_or_slots(self):
        pair=PackedPair();runtime=DistributedRuntime(pair)
        with ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(lambda:list(runtime.generate('model',None,3,raw_prompt=[1]*12))) for _ in range(4)]
            self.queued(runtime,4);_,errors=self.start(runtime)
            results=[future.result(2) for future in futures]
        self.assertEqual([r[-1]['timing']['prefill_batch_size'] for r in results],[2]*4)
        self.assertEqual([c[2] for c in pair.calls if c[0]=='prefill_batch'],[[0,1],[2,3]])
        self.assertEqual(pair.state.lengths,[0]*4);self.assertEqual(errors,[])
