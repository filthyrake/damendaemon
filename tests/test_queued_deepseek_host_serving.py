from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.deepseek_host_generation import HostDeepSeekGenerator
from dllm.queued_deepseek_host_serving import QueuedHostDeepSeekReferenceRuntime
from dllm.runtime import BusyError
from tests import test_deepseek_host_serving as hosts


class QueuedDeepSeekHostServingTests(unittest.TestCase):
    setUp=hosts.DeepSeekHostServingTests.setUp
    session=hosts.DeepSeekHostServingTests.session
    links=hosts.DeepSeekHostServingTests.links
    together=hosts.DeepSeekHostServingTests.together
    pump=hosts.DeepSeekHostServingTests.pump
    serve=hosts.DeepSeekHostServingTests.serve
    post=hosts.DeepSeekHostServingTests.post

    def runtime(self,**options):
        text,workers,control=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=2048)
        runtime=QueuedHostDeepSeekReferenceRuntime(runner,**options);self.addCleanup(runtime.close)
        return runtime,workers,control

    def until(self,condition):
        end=time.monotonic()+3
        while not condition() and time.monotonic()<end:threading.Event().wait(.01)
        self.assertTrue(condition())

    def test_concurrent_actual_http_control_and_authenticated_drain(self):
        runtime,workers,control=self.runtime(queue_size=4);self.serve(runtime)
        with self.assertRaises(ValueError):list(runtime.generate('model',[dict(role='user',content='Hi')],1,temperature=1))
        self.assertTrue(runtime.is_healthy())
        def request():
            with self.post() as response:return json.load(response)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda _:request(),range(3)))
        expected=list(control.generate([dict(role='user',content='Hi')],max_new_tokens=3))
        for result in results:
            self.assertEqual(result['choices'][0]['message'].get('content') or '', ''.join(e.get('content','') for e in expected))
            self.assertEqual(result['usage']['completion_tokens'],expected[-1]['usage']['completion_tokens'])
            self.assertGreaterEqual(result['dllm_timing']['queue_seconds'],0)
        self.assertTrue(all(worker.stage.states==[None,None] for worker in workers))
        with urlopen(Request(self.base+'/v1/scheduler/drain',b'{}',self.admin),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
        with self.assertRaises(HTTPError) as error:self.post()
        self.assertEqual(error.exception.code,503);error.exception.close()
        self.assertTrue(runtime.is_healthy())

    def test_queue_overflow_and_undispatched_timeout_preserve_epoch(self):
        runtime,workers,_=self.runtime(queue_size=1,queue_timeout=.1)
        entered=threading.Event();release=threading.Event();original=workers[-1].stage.forward_layers
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(3):raise RuntimeError('test timeout')
            return original(*args,**kwargs)
        messages=[dict(role='user',content='Hi')]
        with patch.object(workers[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',messages,1)))
            self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',messages,1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            try:
                with self.assertRaisesRegex(BusyError,'full'):list(runtime.generate('model',messages,1))
                with self.assertRaisesRegex(BusyError,'deadline'):second.result(2)
            finally:release.set()
            self.assertTrue(first.result(2)[-1]['finish_reason'])
        self.until(lambda:not runtime.accepted)
        self.assertTrue(runtime.is_healthy());self.assertTrue(list(runtime.generate('model',messages,1)))

    def test_active_abandonment_retires_epoch_and_flushes_waiters(self):
        runtime,_,_=self.runtime(queue_size=2);messages=[dict(role='user',content='Hi')]
        stream=runtime.generate('model',messages,256);next(stream)
        # Stop reading so the bounded output queue holds the generation owner.
        self.until(lambda:next(iter(runtime.accepted.values())).events.qsize()>=8)
        with ThreadPoolExecutor(1) as pool:
            waiting=pool.submit(lambda:list(runtime.generate('model',messages,1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            stream.close()
            with self.assertRaises(BusyError):waiting.result(3)
        self.until(lambda:not runtime.worker.is_alive())
        self.assertFalse(runtime.is_healthy());self.assertTrue(runtime.closed)
        self.assertFalse(runtime.accepted)

    def test_close_interrupts_active_remote_receive_and_flushes_queue(self):
        runtime,workers,_=self.runtime(queue_size=2);messages=[dict(role='user',content='Hi')]
        entered=threading.Event();release=threading.Event();original=workers[-1].stage.forward_layers
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(3):raise RuntimeError('test timeout')
            return original(*args,**kwargs)
        with patch.object(workers[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',messages,2)))
            self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',messages,1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            try:
                runtime.close()
                for future in (first,second):
                    with self.assertRaises((BusyError,RuntimeError,EOFError,OSError,ValueError)):future.result(2)
            finally:release.set()
        self.assertFalse(runtime.worker.is_alive());self.assertFalse(runtime.accepted)
        self.assertTrue(runtime.closed)

    def test_client_turns_and_input_snapshot_are_preserved(self):
        from dllm.request_scope import cache_scope
        runtime,workers,_=self.runtime(queue_size=3);seen=[];entered=threading.Event();release=threading.Event()
        original=workers[0].prefill_chunk
        def prefill(tokens,*args):
            seen.append(list(tokens))
            if len(seen)==1:
                entered.set()
                if not release.wait(3):raise RuntimeError('test timeout')
            return original(tokens,*args)
        alice,bob=object(),object()
        def request(scope,messages):
            token=cache_scope.set(scope)
            try:return list(runtime.generate('model',messages,1))
            finally:cache_scope.reset(token)
        def messages(value):return [dict(role='user',content=value)]
        with patch.object(workers[0],'prefill_chunk',prefill),ThreadPoolExecutor(4) as pool:
            first=pool.submit(request,alice,messages('start'));self.assertTrue(entered.wait(2))
            pending=[];mutable=messages('one')
            for i,(scope,history) in enumerate(((alice,mutable),(alice,messages('two')),(bob,messages('three')))):
                pending.append(pool.submit(request,scope,history))
                self.until(lambda:runtime.requests.qsize()==i+1)
            mutable[0]['content']='changed'
            self.assertFalse(runtime.drain()['drained']);release.set();first.result(3)
            for future in pending:future.result(3)
        self.assertEqual(seen,[runtime.runner.text.chat_tokens(messages(v)) for v in ('start','one','three','two')])
        self.until(lambda:runtime.scheduler_status()['drained']);self.assertTrue(runtime.is_healthy())

    def test_cleanup_failure_retains_owner_after_queue_exit(self):
        runtime,workers,_=self.runtime();messages=[dict(role='user',content='Hi')]
        stream=runtime.generate('model',messages,256);next(stream)
        self.until(lambda:next(iter(runtime.accepted.values())).events.qsize()>=8)
        with patch.object(workers[0].stage,'close',side_effect=RuntimeError('stage cleanup')):
            stream.close();self.until(lambda:not runtime.worker.is_alive())
            self.assertIsNotNone(runtime.failure);self.assertTrue(runtime.cleanup_pending())
            with self.assertRaisesRegex(RuntimeError,'stage cleanup'):runtime.close()
            self.assertIsNotNone(runtime.runner);self.assertFalse(runtime.closed)
        runtime.close();self.assertTrue(runtime.closed);self.assertFalse(runtime.cleanup_pending())

    def test_remote_failure_flushes_waiting_requests_without_reuse(self):
        runtime,workers,_=self.runtime(queue_size=2);messages=[dict(role='user',content='Hi')]
        entered=threading.Event();release=threading.Event()
        def fail(*args,**kwargs):
            entered.set();release.wait(3);raise RuntimeError('remote failed')
        with patch.object(workers[-1].stage,'forward_layers',fail),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',messages,2)));self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',messages,1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1);release.set()
            with self.assertRaises((EOFError,RuntimeError,OSError,ValueError)):first.result(3)
            with self.assertRaises(BusyError):second.result(3)
        self.until(lambda:not runtime.worker.is_alive());self.assertIsNotNone(runtime.failure)
        self.assertTrue(runtime.closed);self.assertFalse(runtime.accepted)
        with self.assertRaises(BusyError):list(runtime.generate('model',messages,1))

    def test_ping_during_output_backpressure_preserves_occupied_slots_and_completion(self):
        runtime,workers,control=self.runtime();messages=[dict(role='user',content='Hi')]
        blocked=threading.Event();resume=threading.Event();publish=runtime._publish;count=0
        def observe(request,event,**options):
            nonlocal count
            if not options.get('terminal',False):count+=1
            # The first consumer has removed exactly one item; publication ten
            # reaches the full eight-event queue after that removal, not before.
            if count==10:
                self.assertEqual(request.events.qsize(),8);blocked.set()
                if not resume.wait(5):raise RuntimeError('test publication barrier timed out')
            return publish(request,event,**options)
        with patch.object(runtime,'_publish',observe):
            stream=runtime.generate('model',messages,32);first=next(stream)
            try:
                self.assertTrue(blocked.wait(2))
                lengths=tuple(workers[0].channel.state.lengths)
                # Peers have a two-second receive bound. Maintenance across a longer
                # pause must preserve occupied slot cursors and exact final output.
                for _ in range(3):
                    threading.Event().wait(.8);workers[0].ping()
                    self.assertEqual(tuple(workers[0].channel.state.lengths),lengths)
            finally:resume.set()
            events=[first,*list(stream)]
        expected=list(control.generate(messages,max_new_tokens=32))
        self.assertEqual(''.join(e.get('text','') for e in events),''.join(e.get('content','') for e in expected))
        self.assertEqual(events[-1]['usage']['completion_tokens'],expected[-1]['usage']['completion_tokens'])
        self.assertTrue(runtime.is_healthy())

    def test_queued_dsml_tools_preserve_native_contract_and_reuse(self):
        from tests.test_deepseek_host_generation import DeepSeekHostGenerationTests,TOOLS
        runtime,workers,_=self.runtime(queue_size=2);self.serve(runtime)
        tokens,_,selection=DeepSeekHostGenerationTests.scripted(self,workers[-1],runtime.runner.text)
        with selection,self.post(tools=TOOLS,max_tokens=len(tokens)) as response:result=json.load(response)
        self.assertEqual(result['choices'][0]['finish_reason'],'tool_calls')
        call=result['choices'][0]['message']['tool_calls'][0]
        self.assertEqual(call['function']['name'],'lookup');self.assertEqual(json.loads(call['function']['arguments']),{'value':'first'})
        with self.post() as response:self.assertEqual(response.status,200)
        self.assertTrue(runtime.is_healthy());self.assertTrue(all(w.stage.states==[None,None] for w in workers))
