"""Final cleanup attempts independent reservations despite a release failure."""
import unittest
from unittest.mock import Mock, patch
from dllm.scheduler import BatchedRuntime, Pending
from dllm.sampling import Sampler, Sampling
from tests.test_scheduler import FakeRuntime, FakePagedExecutor


class PagedCleanupTests(unittest.TestCase):
    def test_cleanup_attempts_all_slots_and_preserves_failures(self):
        for failing in ({0}, {0, 1}):
            with self.subTest(failing=failing):
                # Run the group synchronously; no scheduler worker is launched.
                with patch('dllm.scheduler.Runtime', FakeRuntime), \
                        patch('dllm.scheduler.threading.Thread') as thread:
                    thread.return_value.is_alive.return_value = False
                    runtime = BatchedRuntime({'a': 'a'}, batch_size=2)
                executor = FakePagedExecutor()
                requests = [Pending('a', [{'content': 'ok'}], 2, sampler=Sampler(Sampling())) for _ in range(2)]
                runtime.pending.append(requests[1])
                runtime.requests.update(requests)
                original = executor.release_request
                attempted = []
                def release(slot):
                    attempted.append(slot)
                    if slot in failing: raise RuntimeError(f'failed release {slot}')
                    original(slot)
                executor.release_request = release
                executor.prefill_batch = Mock(side_effect=RuntimeError('failed prefill'))
                try:
                    with self.assertRaises(Exception) as raised:
                        runtime._paged_group(requests[0], executor, 'fake', 0.)
                    self.assertIn(1, attempted)
                    self.assertIsInstance(raised.exception, ExceptionGroup)
                    self.assertEqual({str(error) for error in raised.exception.exceptions},
                                     {f'failed release {slot}' for slot in failing})
                    self.assertTrue(all(request.done.is_set() and request.error for request in requests))
                    for slot in range(2):
                        self.assertEqual(bool(executor.pool.pages[slot]), slot in failing)
                finally:
                    runtime.close()

    def test_unrecovered_release_stops_owner_and_queued_requests(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from dllm.runtime import BusyError
        entered, proceed = threading.Event(), threading.Event()
        executor = FakePagedExecutor()
        def fail_prefill(*args, **kwargs):
            entered.set()
            if not proceed.wait(3):raise RuntimeError('test did not release prefill')
            raise RuntimeError('injected prefill failure')
        executor.prefill_batch = Mock(side_effect=fail_prefill)
        executor.release_request = Mock(side_effect=RuntimeError('injected reservation release failure'))
        with patch('dllm.scheduler.Runtime',FakeRuntime):
            runtime=BatchedRuntime({'a':'a'},batch_size=1,queue_size=3)
        runtime.runtime.executor=executor
        runtime.runtime.close=Mock()
        def generate():return list(runtime.generate('a',[{'content':'ok'}],2))
        with ThreadPoolExecutor(2) as pool:
            try:
                first=pool.submit(generate)
                self.assertTrue(entered.wait(2))
                second=pool.submit(generate)
                with runtime.condition:
                    self.assertTrue(runtime.condition.wait_for(lambda:len(runtime.pending)==1,timeout=2))
                proceed.set()
                with self.assertRaisesRegex(RuntimeError,'injected prefill failure'):first.result(timeout=3)
                with self.assertRaisesRegex(RuntimeError,'scheduler worker stopped'):second.result(timeout=3)
                runtime.worker.join(timeout=2)
                self.assertFalse(runtime.worker.is_alive())
                self.assertTrue(runtime.closed)
                executor.prefill_batch.assert_called_once()
                runtime.runtime.close.assert_not_called()
                with self.assertRaises(BusyError):generate()
            finally:
                proceed.set()
                runtime.close()
        runtime.runtime.close.assert_called_once()
