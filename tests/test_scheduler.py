from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

from dllm.scheduler import BatchedRuntime
from dllm.runtime import BusyError


class Tensor:
    def __init__(self, values):
        self.values = values
    def argmax(self, _):
        return self
    def item(self):
        return self.values[0]
    def tolist(self):
        return self.values


class FakeExecutor:
    context = 256
    kernel_backend = 'fake'
    residual_dtype = 'float16'
    def __init__(self):
        self.model = SimpleNamespace(eos={9}, chat_tokens=lambda messages: [messages[0]['content']],
            tokenizer=SimpleNamespace(decode=lambda tokens, **_: ''.join(map(str, tokens))))
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batches = []
    def prefill(self, tokens, slot):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError('test did not release prefill')
        if tokens[0] == 'fail':
            raise ValueError('invalid prompt')
        return Tensor([9 if tokens[0] == 'eos' else 1])
    def prefill_batch(self, prompts, slots):
        return Tensor([self.prefill(prompt, slot).item() for prompt, slot in zip(prompts, slots)])
    def decode_batch(self, tokens, slots):
        self.batches.append(tuple(slots))
        return None, Tensor([token + 1 for token in tokens])


class FakeRuntime:
    def __init__(self, models, **kwargs):
        self.models = models
        self.executor = FakeExecutor()
        self.groups = []
    def model_ids(self):
        return list(self.models)
    def manage(self, **options):
        self.groups.append('control:'+options['action'])
        return {'done': options['action']}
    @contextmanager
    def session(self, alias):
        self.groups.append(alias)
        yield self.executor, 'disk', 0.01


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('dllm.scheduler.Runtime', FakeRuntime)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime = BatchedRuntime({'a': 'a', 'b': 'b'}, batch_size=2, queue_size=4)
        self.executor = self.runtime.runtime.executor
        self.addCleanup(self.runtime.close)
        self.addCleanup(self.executor.release.set)

    def generate(self, alias='a', prompt='ok', count=5):
        return self.runtime.generate(alias, [{'content': prompt}], count)

    def test_generated_decoder_preserves_continuation_space_in_stream(self):
        self.executor.model.tokenizer.decode_generated = lambda tokens, **_: ' '+''.join(map(str,tokens))
        self.executor.release.set()
        events = list(self.generate(count=3))
        self.assertEqual(''.join(event.get('text','') for event in events), ' 123')

    def wait_pending(self, count):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.runtime.condition:
                if len(self.runtime.pending) == count:
                    return
            time.sleep(.001)
        self.fail('requests did not enter queue')

    def test_management_is_a_barrier_between_generation_groups(self):
        with ThreadPoolExecutor(3) as pool:
            first=pool.submit(lambda:list(self.generate('a',count=4)))
            self.assertTrue(self.executor.entered.wait(2))
            control=pool.submit(self.runtime.manage,'status')
            self.wait_pending(1)
            last=pool.submit(lambda:list(self.generate('a',count=2)))
            self.wait_pending(2)
            self.executor.release.set()
            first.result(3)
            self.assertEqual(control.result(3),{'done':'status'})
            last.result(3)
        self.assertEqual(self.runtime.runtime.groups,['a','control:status','a'])

    def test_management_uses_bounded_admission_and_releases_it(self):
        acquired=[]
        try:
            for _ in range(4):
                self.assertTrue(self.runtime.admission.acquire(blocking=False))
                acquired.append(True)
            with self.assertRaises(BusyError):self.runtime.manage('status')
        finally:
            for _ in acquired:self.runtime.admission.release()
        self.assertEqual(self.runtime.manage('status'),{'done':'status'})
        with self.assertRaises(ValueError):self.runtime.manage('register','a')
        self.assertEqual(self.runtime.manage('status'),{'done':'status'})

    def test_joining_requests_batch_and_report_individual_usage(self):
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(lambda: list(self.generate(count=6)))
            self.assertTrue(self.executor.entered.wait(2))
            second = pool.submit(lambda: list(self.generate(count=3)))
            self.wait_pending(1)
            self.executor.release.set()
            a, b = first.result(3), second.result(3)
        self.assertTrue(any(len(batch) == 2 for batch in self.executor.batches))
        self.assertEqual([x['token'] for x in a[:-1]], list(range(1, 7)))
        self.assertEqual([x['token'] for x in b[:-1]], [1, 2, 3])
        self.assertEqual(b[-1]['timing']['source_tier'], 'gpu')
        self.assertEqual(b[-1]['timing']['activation_seconds'], 0)
        self.assertEqual(b[-1]['usage']['completion_tokens'], 3)

    def test_other_model_gets_turn_before_later_same_model(self):
        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(lambda: list(self.generate('a')))]
            self.assertTrue(self.executor.entered.wait(2))
            futures.append(pool.submit(lambda: list(self.generate('b'))))
            self.wait_pending(1)
            futures.append(pool.submit(lambda: list(self.generate('a'))))
            self.wait_pending(2)
            self.executor.release.set()
            for future in futures:
                future.result(3)
        self.assertEqual(self.runtime.runtime.groups, ['a', 'b', 'a'])

    def test_bad_prompt_and_eos_leave_worker_usable(self):
        self.executor.release.set()
        with self.assertRaises(ValueError):
            list(self.generate(prompt='fail'))
        result = list(self.generate(prompt='eos'))
        self.assertEqual(result[-1]['finish_reason'], 'stop')
        self.assertEqual(result[-1]['usage']['completion_tokens'], 1)
        self.assertEqual(len(list(self.generate(count=2))), 3)

    def test_cancel_backpressured_request_releases_slot(self):
        self.executor.release.set()
        self.executor.model.eos = set()
        stream = self.generate(count=100)
        next(stream)
        stream.close()
        result = list(self.generate('b', count=2))
        self.assertEqual(result[-1]['usage']['completion_tokens'], 2)

    def test_close_wakes_queued_consumers_and_rejects_new_work(self):
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(lambda: list(self.generate()))
            self.assertTrue(self.executor.entered.wait(2))
            second = pool.submit(lambda: list(self.generate('b')))
            self.wait_pending(1)
            closing = pool.submit(self.runtime.close)
            # close must notify consumers even while the GPU worker is occupied.
            with self.assertRaises(BusyError):
                second.result(3)
            self.executor.release.set()
            for future in (first,):
                with self.assertRaises(BusyError):
                    future.result(3)
            closing.result(3)
        self.runtime.close()
        self.assertFalse(self.runtime.worker.is_alive())
        with self.assertRaises(BusyError):
            list(self.generate())

    def test_admission_limit_and_queue_timeout_recover(self):
        self.runtime.queue_timeout = .05
        with ThreadPoolExecutor(4) as pool:
            first = pool.submit(lambda: list(self.generate()))
            self.assertTrue(self.executor.entered.wait(2))
            queued = [pool.submit(lambda: list(self.generate())) for _ in range(3)]
            self.wait_pending(3)
            with self.assertRaises(BusyError):
                list(self.generate())
            time.sleep(.06)
            self.executor.release.set()
            first.result(3)
            for future in queued:
                with self.assertRaises(BusyError):
                    future.result(3)
        self.assertEqual(list(self.generate(count=1))[-1]['usage']['completion_tokens'], 1)

    def test_decode_failure_releases_all_slots(self):
        self.executor.release.set()
        with patch.object(self.executor, 'decode_batch', side_effect=RuntimeError('GPU fault')):
            with self.assertRaisesRegex(RuntimeError, 'GPU fault'):
                list(self.generate())
        self.assertEqual(list(self.generate(count=2))[-1]['usage']['completion_tokens'], 2)

    def test_stalled_stream_times_out_without_blocking_following_model(self):
        self.executor.model.eos = set()
        self.executor.release.set()
        self.runtime.queue_timeout = .03
        stream = self.generate(count=100)
        next(stream)
        # Leave its bounded output queue undrained until the worker expires it.
        time.sleep(.08)
        with self.assertRaisesRegex(BusyError, 'backpressure'):
            list(stream)
        result = list(self.generate('b', count=2))
        self.assertEqual(result[-1]['usage']['completion_tokens'], 2)


class ExecutorLeaseTests(unittest.TestCase):
    def test_worker_releases_previous_executor_before_next_activation(self):
        import gc
        import weakref
        class EphemeralRuntime:
            def __init__(self,models,**kwargs):
                self.models=models
                self.previous=None
            @contextmanager
            def session(self,alias):
                gc.collect()
                if self.previous is not None and self.previous() is not None:
                    raise RuntimeError('previous executor still leased during activation')
                executor=FakeExecutor()
                executor.release.set()
                self.previous=weakref.ref(executor)
                yield executor,'disk',0
        with patch('dllm.scheduler.Runtime',EphemeralRuntime):
            runtime=BatchedRuntime({'a':'a','b':'b'})
            try:
                for alias in ('a','b','a'):
                    result=list(runtime.generate(alias,[{'content':'eos'}],1))
                    self.assertEqual(result[-1]['finish_reason'],'stop')
                retained = None
                try:
                    list(runtime.generate('a',[{'content':'fail'}],1))
                except ValueError as error:
                    retained = error
                self.assertIsNotNone(retained)
                result = list(runtime.generate('b',[{'content':'eos'}],1))
                self.assertEqual(result[-1]['finish_reason'],'stop')
            finally:
                runtime.close()


class ChunkExecutor(FakeExecutor):
    supports_chunked_prefill=True
    def __init__(self):
        super().__init__()
        self.model.chat_tokens=lambda messages: list(messages[0]['content'])
        self.chunk_events=[]
    def prefill_chunk(self,tokens,slot,start,final=True):
        import torch
        self.entered.set()
        if not self.release.wait(5):raise TimeoutError('test did not release chunk')
        if tokens==list('fa'):raise ValueError('invalid prompt')
        self.chunk_events.append(('prefill',slot,start,len(tokens)))
        logits=torch.zeros(1,10)
        logits[0,1]=1
        return logits
    def decode_batch(self,tokens,slots):
        self.chunk_events.append(('decode',tuple(slots)))
        return super().decode_batch(tokens,slots)


class ChunkedSchedulerTests(unittest.TestCase):
    def setUp(self):
        patcher=patch('dllm.scheduler.Runtime',FakeRuntime)
        patcher.start();self.addCleanup(patcher.stop)
        self.runtime=BatchedRuntime({'a':'a','b':'b'},batch_size=2,queue_size=4,prefill_chunk_size=2)
        self.executor=ChunkExecutor()
        self.runtime.runtime.executor=self.executor
        self.addCleanup(self.runtime.close)
        self.addCleanup(self.executor.release.set)

    def generate(self,prompt,count=8,alias='a'):
        return list(self.runtime.generate(alias,[{'content':prompt}],count))

    def wait_pending(self,count):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            with self.runtime.condition:
                if len(self.runtime.pending)==count:return
            time.sleep(.001)
        self.fail('requests did not queue')

    def test_decode_interleaves_chunks_and_preserves_usage(self):
        with ThreadPoolExecutor(2) as pool:
            short=pool.submit(self.generate,'ab')
            self.assertTrue(self.executor.entered.wait(2))
            long=pool.submit(self.generate,'abcdefgh')
            self.wait_pending(1)
            self.executor.release.set()
            a,b=short.result(3),long.result(3)
        chunks=[i for i,e in enumerate(self.executor.chunk_events) if e[0]=='prefill' and e[1]==1]
        self.assertEqual([self.executor.chunk_events[i][2] for i in chunks],[0,2,4,6])
        for a_index,b_index in zip(chunks,chunks[1:]):
            self.assertTrue(any(e[0]=='decode' and 0 in e[1] for e in self.executor.chunk_events[a_index+1:b_index]))
        self.assertEqual(b[-1]['usage']['prompt_tokens'],8)
        self.assertEqual(a[-1]['usage']['prompt_tokens'],2)
        self.assertEqual([e['token'] for e in b if 'token' in e],list(range(1,9)))

    def test_partial_group_preserves_management_barrier(self):
        with ThreadPoolExecutor(3) as pool:
            first=pool.submit(self.generate,'abcdef',2)
            self.assertTrue(self.executor.entered.wait(2))
            control=pool.submit(self.runtime.manage,'status')
            self.wait_pending(1)
            last=pool.submit(self.generate,'ab',2)
            self.wait_pending(2)
            self.executor.release.set()
            first.result(3);control.result(3);last.result(3)
        self.assertEqual(self.runtime.runtime.groups,['a','control:status','a'])

    def test_partial_cancellation_reuses_slot_without_emitting(self):
        with ThreadPoolExecutor(1) as pool:
            cancelled=pool.submit(self.generate,'abcdefgh')
            self.assertTrue(self.executor.entered.wait(2))
            with self.runtime.condition:
                request=next(iter(self.runtime.requests))
                request.cancelled.set()
            self.executor.release.set()
            self.assertEqual(cancelled.result(3),[])
        result=self.generate('ab',2)
        self.assertEqual(result[-1]['usage']['completion_tokens'],2)
        self.assertEqual([e[2] for e in self.executor.chunk_events if e[0]=='prefill'],[0,0])

    def test_unsupported_executor_keeps_full_prefill_path(self):
        self.executor.supports_chunked_prefill=False
        self.executor.release.set()
        result=self.generate('ab',2)
        self.assertEqual(result[-1]['usage']['completion_tokens'],2)
        self.assertFalse(any(e[0]=='prefill' for e in self.executor.chunk_events))

    def test_invalid_chunk_budget_is_rejected(self):
        for value in (-1,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                BatchedRuntime({},prefill_chunk_size=value)


class FakePagedExecutor(FakeExecutor):
    def __init__(self):
        from dllm.pages import PagePool
        super().__init__()
        self.pool=PagePool(128,2,self.context)
        self.deferred=0
    def reserve_request(self,slot,tokens):
        from dllm.pages import PageCapacityError
        if self.pool.needed(tokens)>self.pool.capacity:
            raise ValueError('request exceeds pool')
        try:
            self.pool.resize_many([(slot,tokens)],replace=True)
        except PageCapacityError:
            self.deferred+=1
            raise
    def release_request(self,slot):
        self.pool.release(slot)


class PagedSchedulerTests(SchedulerTests):
    def setUp(self):
        super().setUp()
        self.executor=FakePagedExecutor()
        self.runtime.runtime.executor=self.executor
        self.addCleanup(self.executor.release.set)

    def test_capacity_waits_then_reuses_pages(self):
        from dllm.pages import PagePool
        self.executor.pool=PagePool(32,2,self.executor.context)
        with ThreadPoolExecutor(3) as pool:
            first=pool.submit(lambda:list(self.generate(count=20)))
            self.assertTrue(self.executor.entered.wait(2))
            second=pool.submit(lambda:list(self.generate(count=20)))
            third=pool.submit(lambda:list(self.generate(count=20)))
            self.wait_pending(2)
            self.executor.release.set()
            for future in (first,second,third):
                self.assertIn('finish_reason',future.result(timeout=5)[-1])
        self.runtime.manage('status')
        self.assertGreater(self.executor.deferred,0)
        self.assertEqual(self.executor.pool.used_tokens,0)

    def test_failed_prefill_releases_reservation(self):
        self.executor.release.set()
        with self.assertRaises(ValueError):
            list(self.generate(prompt='fail',count=20))
        self.assertIn('finish_reason',list(self.generate(count=20))[-1])
        self.runtime.manage('status')
        self.assertEqual(self.executor.pool.used_tokens,0)

    def test_impossible_request_rejected_without_leak(self):
        self.executor.release.set()
        with self.assertRaises(ValueError):
            list(self.generate(count=200))
        self.runtime.manage('status')
        self.assertEqual(self.executor.pool.used_tokens,0)


class PagedChunkExecutor(ChunkExecutor,FakePagedExecutor):
    pass


class PagedChunkedSchedulerTests(ChunkedSchedulerTests):
    def setUp(self):
        super().setUp()
        self.executor=PagedChunkExecutor()
        self.runtime.runtime.executor=self.executor
        self.addCleanup(self.executor.release.set)

    def tearDown(self):
        self.runtime.manage('status')
        self.assertEqual(self.executor.pool.used_tokens,0)

    def test_chunk_error_returns_pages_and_recovers(self):
        self.executor.release.set()
        with self.assertRaises(ValueError):
            self.generate('fail')
        self.assertEqual(self.generate('ab',2)[-1]['usage']['completion_tokens'],2)

    def test_chunk_capacity_waits_for_prior_owner(self):
        from dllm.pages import PagePool
        self.executor.pool=PagePool(32,2,self.executor.context)
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(self.generate,'abcdefghijklmnop',8)
            self.assertTrue(self.executor.entered.wait(2))
            second=pool.submit(self.generate,'abcdefghijklmnop',8)
            self.wait_pending(1)
            self.executor.release.set()
            for future in (first,second):
                self.assertEqual(future.result(5)[-1]['usage']['completion_tokens'],8)
        self.assertGreater(self.executor.deferred,0)

    def test_exact_prefix_hit_still_reserves_and_releases_pages(self):
        import torch
        from unittest.mock import Mock
        logits=torch.zeros(1,10);logits[0,1]=1
        cache=SimpleNamespace(restore=Mock(return_value=(2,logits)),remember=Mock())
        self.runtime.prefix_cache_tokens=32
        with patch('dllm.prefix.PrefixCache',return_value=cache):
            result=self.generate('ab',2)
        self.assertEqual(result[-1]['timing']['cached_prompt_tokens'],2)
        self.assertFalse(any(e[0]=='prefill' for e in self.executor.chunk_events))
        cache.remember.assert_not_called()

    def test_prefix_restore_failure_returns_reservation(self):
        from unittest.mock import Mock
        cache=SimpleNamespace(restore=Mock(side_effect=[ValueError('bad snapshot'),(0,None)]),remember=Mock())
        self.runtime.prefix_cache_tokens=32
        self.executor.release.set()
        with patch('dllm.prefix.PrefixCache',return_value=cache):
            with self.assertRaisesRegex(ValueError,'bad snapshot'):
                self.generate('ab',2)
            self.assertEqual(self.generate('ab',2)[-1]['usage']['completion_tokens'],2)


class CompletionRaceTests(unittest.TestCase):
    def test_final_block_arriving_after_empty_observation_is_drained(self):
        from collections import deque
        import queue
        from dllm.scheduler import Pending
        runtime=BatchedRuntime.__new__(BatchedRuntime)
        runtime.models={'a':'model'};runtime.admission=threading.BoundedSemaphore(1)
        runtime.condition=threading.Condition();runtime.pending=deque();runtime.requests=set();runtime.closed=False
        request=Pending('a',[],1)
        class RacingQueue(queue.Queue):
            raced=False
            def get_nowait(self):
                if not self.raced:
                    self.raced=True
                    # Simulate the producer finishing after the consumer's
                    # failed nonblocking read but before it checks done.
                    self.put_nowait({'token':17,'text':'x'})
                    request.final={'finish_reason':'length'}
                    request.done.set();request.ready.set()
                    raise queue.Empty
                return super().get_nowait()
        request.events=RacingQueue(maxsize=8)
        with patch('dllm.scheduler.Pending',return_value=request):
            events=list(runtime.generate('a',[{'role':'user','content':'x'}],1))
        self.assertEqual(events,[{'token':17,'text':'x'},{'finish_reason':'length'}])
