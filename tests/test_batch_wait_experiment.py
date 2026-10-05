from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.experiments.batch_wait import wait_for_batch


def request(alias='a',command=None):
    return SimpleNamespace(alias=alias,command=command,done=threading.Event(),cancelled=threading.Event())


class BatchWaitExperimentTests(unittest.TestCase):
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
        with patch('scripts.experiments.batch_wait.time.monotonic',return_value=0):
            wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_called_once_with(timeout=.001)
        self.assertEqual(list(runtime.pending),[incoming])

    def test_deadline_close_cancel_and_single_slot_do_not_wait(self):
        runtime=self.runtime()
        with patch('scripts.experiments.batch_wait.time.monotonic',side_effect=[0,.002]):
            wait_for_batch(runtime,request(),.001)
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
        with patch('scripts.experiments.batch_wait.time.monotonic',return_value=0):
            wait_for_batch(runtime,request(),.001)
        runtime.condition.wait.assert_called_once()
        self.assertEqual(list(runtime.pending),[finished,cancelled,incoming])


if __name__=='__main__':unittest.main()
