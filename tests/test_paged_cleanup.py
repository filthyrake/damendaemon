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
