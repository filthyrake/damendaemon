from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import unittest
from dllm.distributed_admission import DistributedAdmission
from dllm.distributed_serving import Request
from dllm.request_scope import CacheScope,cache_scope


def request(scope,token):
    return Request([token],1,time.monotonic(),queue.Queue(9),cache_scope=scope)


class DistributedAdmissionTests(unittest.TestCase):
    def test_unequal_client_bursts_preserve_fifo_and_bounded_identity_history(self):
        pending=DistributedAdmission(10,60);a,b,c=CacheScope(),CacheScope(),CacheScope()
        lanes=[[request(scope,n) for n in range(count)] for scope,count in ((a,5),(b,3),(c,2))]
        for item in sum(lanes,[]):pending.put_nowait(item)
        actual=[]
        while not pending.empty():
            actual.append(pending.get_nowait());self.assertLessEqual(len(pending.fair.recent),3)
        expected=[lanes[0][0],lanes[1][0],lanes[2][0],lanes[0][1],lanes[1][1],lanes[2][1],lanes[0][2],lanes[1][2],lanes[0][3],lanes[0][4]]
        self.assertEqual([id(x) for x in actual],[id(x) for x in expected]);self.assertFalse(pending.fair.recent)
        anonymous=[request(None,n) for n in range(6)]
        for item in anonymous:pending.put_nowait(item)
        self.assertEqual([id(pending.get_nowait()) for _ in anonymous],[id(x) for x in anonymous])

    def test_stale_entries_do_not_consume_turns_or_escape_capacity(self):
        pending=DistributedAdmission(5,1);a,b=CacheScope(),CacheScope()
        first,second=request(a,1),request(a,2)
        pending.put(first);pending.put(second);self.assertIs(pending.get(),first)
        expired=request(b,3);expired.submitted-=2;cancelled=request(b,4);cancelled.cancelled.set()
        live=request(b,5)
        for item in (expired,cancelled,live):pending.put(item)
        self.assertIs(pending.get(),expired);self.assertIs(pending.get(),cancelled)
        self.assertIs(pending.get(),live);self.assertIs(pending.get(),second)
        self.assertFalse(pending.fair.recent)

    def test_queue_mutex_capacity_producer_wakeup_and_context_capture(self):
        pending=DistributedAdmission(1,60);a,b=CacheScope(),CacheScope();first=request(a,1);second=request(b,2)
        pending.put(first)
        with self.assertRaises(queue.Full):pending.put_nowait(second)
        entered=threading.Event()
        def produce():entered.set();pending.put(second,timeout=2)
        with ThreadPoolExecutor(1) as pool:
            producer=pool.submit(produce);self.assertTrue(entered.wait(1));self.assertFalse(producer.done())
            self.assertIs(pending.get(),first);producer.result(2)
        self.assertIs(pending.get(),second);self.assertEqual(pending.qsize(),0)
        token=cache_scope.set(a)
        try:captured=Request([1],1,time.monotonic(),queue.Queue(9))
        finally:cache_scope.reset(token)
        self.assertIs(captured.cache_scope,a);self.assertNotIn('cache_scope',repr(captured))
