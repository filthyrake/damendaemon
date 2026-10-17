"""Unexpected worker exit closes admission and wakes every retained caller."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import Mock, patch
from damendaemon.runtime import BusyError
from damendaemon.scheduler import BatchedRuntime
from tests.test_scheduler import FakeRuntime


class SchedulerWorkerFailureTests(unittest.TestCase):
    def test_fatal_generation_and_command_wake_owned_and_queued_callers(self):
        for operation in ('generation', 'command'):
            with self.subTest(operation=operation):
                with patch('damendaemon.scheduler.Runtime', FakeRuntime):
                    runtime = BatchedRuntime({'a': 'a'}, batch_size=1, queue_size=3)
                runtime.runtime.close = Mock()
                entered, release = threading.Event(), threading.Event()
                def fail(*args, **kwargs):
                    entered.set()
                    if not release.wait(3): raise RuntimeError('test did not release worker')
                    raise SystemExit('injected worker exit')
                target = runtime.runtime.executor if operation == 'generation' else runtime.runtime
                method = 'prefill_batch' if operation == 'generation' else 'manage'
                def generate(): return list(runtime.generate('a', [{'content': 'ok'}], 1))
                with patch.object(target, method, side_effect=fail), \
                        patch('threading.excepthook'), ThreadPoolExecutor(2) as pool:
                    first = pool.submit(generate if operation == 'generation' else lambda: runtime.manage('status'))
                    second = None
                    try:
                        self.assertTrue(entered.wait(2))
                        second = pool.submit(generate)
                        with runtime.condition:
                            self.assertTrue(runtime.condition.wait_for(lambda: len(runtime.pending)==1, timeout=2))
                        release.set()
                        for future in (first, second):
                            with self.assertRaisesRegex(RuntimeError, 'worker stopped'): future.result(1)
                        runtime.worker.join(2)
                        self.assertFalse(runtime.worker.is_alive())
                        self.assertFalse(runtime.is_healthy())
                        self.assertTrue(runtime.scheduler_status()['closed'])
                        self.assertFalse(runtime.requests)
                        self.assertFalse(runtime.pending)
                        runtime.runtime.close.assert_not_called()
                        with self.assertRaises(BusyError): generate()
                    finally:
                        release.set()
                        # Ensure a failing regression cannot strand the test pool.
                        with runtime.condition:
                            for request in tuple(runtime.requests):
                                runtime._finish(request, RuntimeError('test cleanup'))
                        runtime.close()
                runtime.runtime.close.assert_called_once_with()
