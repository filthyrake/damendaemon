"""Failure status must not own dead worker stack allocations."""
import gc
import queue
import time
import unittest
import weakref

from damendaemon.distributed_serving import DistributedRuntime, Request
from tests.test_distributed_serving import Pair


class DistributedFailureMemoryTests(unittest.TestCase):
    def test_failure_status_does_not_retain_worker_frames(self):
        for stage in ('heartbeat','release','execution-and-release'):
            with self.subTest(stage=stage):
                references=[]
                class Allocation:pass
                def fail(message):
                    allocation=Allocation();references.append(weakref.ref(allocation))
                    error=ValueError(message);error.add_note('fixture note')
                    raise error from RuntimeError('cause fixture')
                class FailingPair(Pair):
                    def heartbeat(self):fail('heartbeat')
                    def step(self,tokens):
                        runtime.close()
                        if stage=='execution-and-release':fail('execution')
                        return super().step(tokens)
                    def release(self):fail('release')
                runtime=DistributedRuntime(FailingPair())
                if stage!='heartbeat':
                    runtime.requests.put_nowait(Request([1],2,time.monotonic(),queue.Queue()))
                expected='execution' if stage=='execution-and-release' else stage
                try:
                    runtime.run(heartbeat_interval=.001)
                except ValueError as original:
                    self.assertEqual(str(original),expected)
                    self.assertIsNotNone(original.__traceback__)
                    self.assertIsNotNone(original.__cause__)
                    self.assertIsNot(runtime.failure,original)
                    if stage=='execution-and-release':
                        self.assertTrue(any('slot release failed' in note for note in original.__notes__))
                else:self.fail('worker error was hidden')
                gc.collect()
                self.assertTrue(references)
                self.assertTrue(all(ref() is None for ref in references))
                self.assertIsInstance(runtime.failure,ValueError)
                self.assertEqual(str(runtime.failure),expected)
                self.assertIn('fixture note',runtime.failure.__notes__)
                self.assertIsNone(runtime.failure.__traceback__)
                self.assertIsNone(runtime.failure.__cause__)
                self.assertIsNone(runtime.failure.__context__)
                self.assertTrue(runtime.scheduler_status()['failed'])
                self.assertTrue(runtime.closed.is_set())
                self.assertFalse(runtime.execution_lock.locked())
