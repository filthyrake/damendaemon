from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from dllm.queued_reference_serving import QueuedReferenceRuntime,BUFFER_EVENTS
from dllm.request_scope import cache_scope
from dllm.runtime import BusyError
from tests import test_reference_serving as fixtures


def wait_for(predicate):
    deadline=time.monotonic()+3
    while not predicate() and time.monotonic()<deadline:time.sleep(.001)
    if not predicate():raise AssertionError('queue condition did not become true')


class QueuedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReferenceServingTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def runtime(self,family='glm',**options):
        with patch.object(fixtures,'ReferenceRuntime',partial(QueuedReferenceRuntime,**options)):
            return self.fixture.runtime(family,tools=True)

    def generate(self,runtime,**options):
        return runtime.generate('model',self.fixture.messages,512,tools=fixtures.TOOLS,**options)

    def test_invalid_owned_streams_fail_without_hanging_or_publishing_success(self):
        for case in ('empty', 'missing', 'duplicate', 'late', 'cleanup', 'invalid-final'):
            with self.subTest(case=case):
                runtime, _ = self.runtime()
                observed = []
                def owned(request):
                    try:
                        if case == 'empty': return
                        yield {'text': 'partial'}
                        if case == 'missing': return
                        final = dict(finish_reason='stop', usage={}, timing={})
                        yield dict(final, finish_reason=None) if case == 'invalid-final' else final
                        if case == 'duplicate': yield final
                        if case == 'late': yield {'text': 'late'}
                    finally:
                        if case == 'cleanup': raise RuntimeError('injected iterator cleanup failure')
                def consume():
                    for event in self.generate(runtime): observed.append(event)
                with patch.object(runtime, '_generate_owned', side_effect=owned), ThreadPoolExecutor(1) as pool:
                    future = pool.submit(consume)
                    try:
                        with self.assertRaises(RuntimeError): future.result(1)
                        self.assertFalse(any('finish_reason' in event for event in observed))
                    finally: runtime.close()

    def test_terminal_publication_waits_for_owned_iterator_cleanup(self):
        runtime, _ = self.runtime()
        entered, release = threading.Event(), threading.Event()
        def owned(request):
            try:
                yield dict(finish_reason='stop', usage={}, timing={})
            finally:
                entered.set()
                if not release.wait(3): raise RuntimeError('test did not release cleanup')
        with patch.object(runtime, '_generate_owned', side_effect=owned), ThreadPoolExecutor(1) as pool:
            future = pool.submit(lambda: list(self.generate(runtime)))
            try:
                self.assertTrue(entered.wait(2))
                from concurrent.futures import TimeoutError as FutureTimeout
                with self.assertRaises(FutureTimeout): future.result(.05)
            finally: release.set()
            self.assertEqual(future.result(2)[-1]['finish_reason'], 'stop')
        runtime.close()

    def test_concurrent_http_preserves_family_tools_and_reasoning(self):
        for family in ('glm','flash','deepseek'):
            runtime,model=self.runtime(family);base=self.fixture.serve(runtime)
            def request():
                with self.fixture.post(base,self.fixture.body(family,tools=fixtures.TOOLS)) as response:return json.load(response)
            with ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda _:request(),range(3)))
            ids=[]
            for result in results:
                message=result['choices'][0]['message'];self.assertEqual(message['reasoning_content'],'reason')
                self.assertEqual(message['content'].strip(),'answer');self.assertEqual(result['choices'][0]['finish_reason'],'tool_calls')
                ids.append(message['tool_calls'][0]['id']);self.assertGreaterEqual(result['dllm_timing']['queue_seconds'],0)
            self.assertEqual(len(set(ids)),3);self.assertEqual(len(model.states),3)
            self.assertTrue(all(s.closed for s in model.states))
            with urlopen(base+'/v1/scheduler/status') as response:self.assertEqual(json.load(response)['batch_capacity'],1)
            runtime.close();self.assertFalse(runtime.worker.is_alive())

    def test_fair_order_snapshot_capacity_and_drain(self):
        runtime,model=self.runtime(queue_size=3)
        entered=threading.Event();resume=threading.Event();seen=[];original=model.prefill
        def prefill(tokens,**options):
            seen.append(tokens.tolist())
            if len(seen)==1:entered.set();self.assertTrue(resume.wait(3))
            return original(tokens,**options)
        alice,bob=object(),object()
        def request(scope,content):
            marker=cache_scope.set(scope)
            try:return list(runtime.generate('model',[dict(role='user',content=content)],512,tools=fixtures.TOOLS))
            finally:cache_scope.reset(marker)
        with patch.object(model,'prefill',side_effect=prefill),ThreadPoolExecutor(4) as pool:
            first=pool.submit(request,alice,'start');self.assertTrue(entered.wait(3))
            pending=[]
            for i,(scope,text) in enumerate(((alice,'one'),(alice,'two'),(bob,'three'))):
                pending.append(pool.submit(request,scope,text))
                wait_for(lambda:runtime.requests.qsize()==i+1)
            with self.assertRaisesRegex(BusyError,'full'):list(self.generate(runtime))
            self.assertFalse(runtime.drain()['drained'])
            with self.assertRaisesRegex(BusyError,'draining'):list(self.generate(runtime))
            resume.set();first.result(5)
            for future in pending:future.result(5)
        expected=['start','one','three','two']
        self.assertEqual(seen,[[runtime.runner.text.tool_chat_tokens([dict(role='user',content=value)],fixtures.TOOLS)] for value in expected])
        wait_for(lambda:runtime.scheduler_status()['drained'])
        self.assertTrue(all(state.closed for state in model.states))

    def test_expired_waiter_reclaims_capacity_while_owner_is_busy(self):
        runtime,model=self.runtime(queue_size=1,queue_timeout=.1)
        entered=threading.Event();resume=threading.Event();original=model.prefill;seen=[]
        def prefill(tokens,**options):
            seen.append(tokens.tolist())
            if len(seen)==1:
                entered.set();self.assertTrue(resume.wait(3))
            return original(tokens,**options)
        with patch.object(model,'prefill',side_effect=prefill),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(self.generate(runtime)))
            try:
                self.assertTrue(entered.wait(3))
                expired=pool.submit(lambda:list(self.generate(runtime)))
                with self.assertRaisesRegex(BusyError,'deadline'):expired.result(3)
                self.assertEqual(runtime.requests.qsize(),0)
                self.assertEqual(len(runtime.accepted),1)
                replacement=pool.submit(lambda:list(self.generate(runtime)))
                wait_for(lambda:runtime.requests.qsize()==1)
                self.assertFalse(first.done())
                resume.set()
                first.result(5);replacement.result(5)
            finally:resume.set()
        wait_for(lambda:not runtime.accepted)
        self.assertEqual(len(seen),2)
        runtime.close()

    def test_queue_expiry_never_prefills_and_mutation_does_not_change_admitted_input(self):
        runtime,model=self.runtime(queue_timeout=.1)
        entered=threading.Event();resume=threading.Event();original=model.prefill;seen=[]
        def prefill(tokens,**options):
            seen.append(tokens.tolist());entered.set();self.assertTrue(resume.wait(3));return original(tokens,**options)
        with patch.object(model,'prefill',side_effect=prefill),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(self.generate(runtime)));self.assertTrue(entered.wait(3))
            expired=pool.submit(lambda:list(self.generate(runtime)))
            with self.assertRaisesRegex(BusyError,'deadline'):expired.result(3)
            resume.set();first.result(5)
        wait_for(lambda:not runtime.accepted);self.assertEqual(len(seen),1)
        runtime.close();runtime,model=self.runtime(queue_timeout=2)
        # Hold one stream under backpressure while another snapshots caller data.
        stream=self.generate(runtime);next(stream)
        wait_for(lambda:next(iter(runtime.accepted.values())).events.qsize()==BUFFER_EVENTS)
        messages=[dict(role='user',content='original')]
        with ThreadPoolExecutor(1) as pool:
            waiting=pool.submit(lambda:list(runtime.generate('model',messages,512,tools=fixtures.TOOLS)))
            wait_for(lambda:runtime.requests.qsize()==1);messages[0]['content']='changed'
            queued=list(runtime.accepted.values())[-1]
            self.assertEqual(queued.messages[0]['content'],'original')
            stream.close();waiting.result(5)
        self.assertTrue(all(s.closed for s in model.states))

    def test_abandoned_backpressure_releases_before_next_and_close_joins(self):
        runtime,model=self.runtime()
        stream=self.generate(runtime);next(stream)
        wait_for(lambda:next(iter(runtime.accepted.values())).events.qsize()==BUFFER_EVENTS)
        self.assertLessEqual(len(model.states),1)
        stream.close()
        result=list(self.generate(runtime));self.assertEqual(result[-1]['finish_reason'],'tool_calls')
        self.assertTrue(all(s.closed for s in model.states))
        runtime.close();self.assertFalse(runtime.worker.is_alive());self.assertTrue(runtime.closed)
        with self.assertRaises(BusyError):list(self.generate(runtime))

    def test_cleanup_failure_closes_admission_and_retains_retry(self):
        runtime,model=self.runtime()
        stream=self.generate(runtime);next(stream)
        wait_for(lambda:next(iter(runtime.accepted.values())).events.qsize()==BUFFER_EVENTS)
        state=model.states[0]
        with patch.object(state,'close',side_effect=RuntimeError('cleanup failed')):
            stream.close();wait_for(runtime.stopping.is_set)
            self.assertIsNotNone(runtime.failure);self.assertTrue(runtime.cleanup_pending())
            with self.assertRaises(BusyError):list(self.generate(runtime))
            with self.assertRaisesRegex(RuntimeError,'cleanup'):runtime.close()
        runtime.close();self.assertTrue(state.closed);self.assertTrue(runtime.closed)

    def test_bad_options_do_not_poison_owner_and_shutdown_wakes_waiters(self):
        runtime,model=self.runtime()
        with self.assertRaises(ValueError):list(self.generate(runtime,temperature=1))
        self.assertFalse(model.states);self.assertFalse(runtime.stopping.is_set())
        stream=self.generate(runtime);next(stream)
        wait_for(lambda:next(iter(runtime.accepted.values())).events.qsize()==BUFFER_EVENTS)
        with ThreadPoolExecutor(1) as pool:
            queued=pool.submit(lambda:list(self.generate(runtime)))
            wait_for(lambda:runtime.requests.qsize()==1);runtime.close()
            with self.assertRaises(BusyError):queued.result(3)
        stream.close();self.assertEqual(runtime.accepted,{});self.assertFalse(runtime.worker.is_alive())

    def test_actual_flash_images_use_the_queued_owner(self):
        import torch
        from scripts import serve_cpu_reference as command
        from tests.test_glm_flash_inline_images import snapshot,messages
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        path=self.fixture.path/'images';path.mkdir();snapshot(path)
        args=command.parser_for_command().parse_args(['flash',str(path),'--flash-images','--context','256','--queue-size','2'])
        runtime,_=command.prepare(args);self.addCleanup(runtime.close);base=self.fixture.serve(runtime)
        def request():
            with self.fixture.post(base,dict(model='model',messages=messages(),max_tokens=2)) as response:return json.load(response)
        with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:request(),range(2)))
        self.assertEqual([r['usage']['completion_tokens'] for r in results],[2,2])
        self.assertEqual(results[0]['choices'],results[1]['choices'])
        self.assertTrue(runtime.manage('status')['queue_enabled']);self.assertTrue(runtime.images)
        runtime.close();self.assertFalse(runtime.worker.is_alive())

    def test_close_during_input_snapshot_cannot_enqueue_after_exit(self):
        runtime,_=self.runtime();entered=threading.Event();resume=threading.Event()
        dumps=json.dumps
        def snapshot(*args,**kwargs):
            entered.set();self.assertTrue(resume.wait(3));return dumps(*args,**kwargs)
        with patch('dllm.queued_reference_serving.json.dumps',side_effect=snapshot),ThreadPoolExecutor(1) as pool:
            result=pool.submit(lambda:list(self.generate(runtime)));self.assertTrue(entered.wait(3))
            runtime.close();resume.set()
            with self.assertRaises(BusyError):result.result(3)
        self.assertTrue(runtime.requests.empty());self.assertEqual(runtime.accepted,{})
