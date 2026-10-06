from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dllm.admission import wait_for_batch


def request(alias='a',command=None):
    return SimpleNamespace(alias=alias,command=command,done=threading.Event(),cancelled=threading.Event())


class AdmissionTests(unittest.TestCase):
    def runtime(self,*pending,batch_size=2):
        return SimpleNamespace(batch_size=batch_size,closed=False,pending=deque(pending),condition=MagicMock())

    def test_ready_batch_and_fifo_barriers_never_wait(self):
        for pending in [(request(),),(request('b'),),(request(command={'action':'status'}),)]:
            runtime=self.runtime(*pending)
            wait_for_batch(runtime,request(),.001)
            runtime.condition.wait.assert_not_called()
            self.assertEqual(list(runtime.pending),list(pending))

    def test_new_arrival_wakes_without_consuming_or_reordering_requests(self):
        runtime=self.runtime();incoming=request()
        runtime.condition.wait.side_effect=lambda **_:runtime.pending.append(incoming)
        with patch('dllm.admission.time.monotonic',return_value=0):
            wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_called_once_with(timeout=.001)
        self.assertEqual(list(runtime.pending),[incoming])

    def test_deadline_close_cancel_and_single_slot_do_not_wait(self):
        runtime=self.runtime()
        with patch('dllm.admission.time.monotonic',side_effect=[0,.002]):
            wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_not_called()
        wait_for_batch(runtime,request(),0)
        runtime.condition.wait.assert_not_called()
        runtime.closed=True;wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_not_called()
        runtime.closed=False;first=request();first.cancelled.set()
        wait_for_batch(runtime,first,.001);runtime.condition.wait.assert_not_called()
        runtime.batch_size=1;wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_not_called()

    def test_finished_entries_do_not_count_toward_batch_capacity(self):
        finished=request();finished.done.set();cancelled=request();cancelled.cancelled.set()
        runtime=self.runtime(finished,cancelled);incoming=request()
        runtime.condition.wait.side_effect=lambda **_:runtime.pending.append(incoming)
        with patch('dllm.admission.time.monotonic',return_value=0):
            wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_called_once()
        self.assertEqual(list(runtime.pending),[finished,cancelled,incoming])

    def test_real_condition_wakes_for_arrival_cancellation_and_shutdown(self):
        class ObservedCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)

        for action in ('arrival', 'barrier', 'cancel', 'close'):
            with self.subTest(action=action):
                waiting=threading.Event()
                first=request()
                runtime=self.runtime()
                runtime.condition=ObservedCondition()
                incoming=request('b' if action=='barrier' else 'a')
                worker=threading.Thread(target=wait_for_batch,args=(runtime,first,10))
                worker.start()
                try:
                    self.assertTrue(waiting.wait(timeout=2))
                    with runtime.condition:
                        if action in ('arrival','barrier'):runtime.pending.append(incoming)
                        elif action=='cancel':first.cancelled.set()
                        else:runtime.closed=True
                        runtime.condition.notify_all()
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive(), 'notification did not release admission wait')
                    self.assertEqual(list(runtime.pending),[incoming] if action in ('arrival','barrier') else [])
                finally:
                    with runtime.condition:
                        runtime.closed=True
                        runtime.condition.notify_all()
                    worker.join(timeout=2)


if __name__=='__main__':unittest.main()


class QueueLimitTests(unittest.TestCase):
    def test_invalid_limits_fail_before_gpu_initialization(self):
        from dllm.runtime import Runtime
        from dllm.scheduler import BatchedRuntime
        from unittest.mock import patch
        invalid = [(value, 120) for value in (True, 1.5, 0, -1, None)]
        invalid += [(8, value) for value in (True, 0, -1, float('nan'), float('inf'), 1e100, 10**1000, None)]
        for constructor in (Runtime, BatchedRuntime):
            for capacity, timeout in invalid:
                with self.subTest(constructor=constructor, capacity=capacity, timeout=timeout), \
                        patch('dllm.runtime.require_gpu') as gpu, self.assertRaises(ValueError):
                    constructor({}, queue_size=capacity, queue_timeout=timeout)
                gpu.assert_not_called()

    def test_scheduler_forwards_limits_to_inner_runtime(self):
        from dllm.scheduler import BatchedRuntime
        from unittest.mock import patch
        with patch('dllm.scheduler.Runtime') as inner:
            runtime = BatchedRuntime({}, batch_size=2, queue_size=6, queue_timeout=.25)
            try:
                self.assertEqual(inner.call_args.kwargs['queue_size'], 6)
                self.assertEqual(inner.call_args.kwargs['queue_timeout'], .25)
                self.assertEqual(runtime.scheduler_status()['queue_timeout_seconds'], .25)
            finally:
                runtime.close()
