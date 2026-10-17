from collections import deque
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch
from damendaemon.fair_admission import FairAdmission
from damendaemon.scheduler import BatchedRuntime,Pending
from damendaemon.request_scope import CacheScope
from damendaemon.runtime import BusyError


def runtime():
    r=BatchedRuntime.__new__(BatchedRuntime)
    r.condition=threading.Condition();r.pending=deque();r.requests=set();r.queue_timeout=60
    r.fair_admission=FairAdmission()
    return r


def request(scope,alias='a',command=None):
    return Pending(alias,[],1,cache_scope=scope,command=command)


class FairAdmissionTests(unittest.TestCase):
    def test_unequal_bursts_rotate_three_lanes_preserving_each_fifo(self):
        scopes=[CacheScope() for _ in range(3)];r=runtime()
        lanes=[[request(scope) for _ in range(count)] for scope,count in zip(scopes,(5,3,2))]
        r.pending.extend(sum(lanes,[]));actual=[]
        with r.condition:
            while r.pending:actual.append(r._next())
            self.assertIsNone(r._next())
        self.assertEqual(actual,[lanes[0][0],lanes[1][0],lanes[2][0],lanes[0][1],lanes[1][1],lanes[2][1],lanes[0][2],lanes[1][2],lanes[0][3],lanes[0][4]])
        self.assertEqual(r.fair_admission.recent,{})

    def test_new_waiting_lane_precedes_continuing_burst_and_history_is_bounded(self):
        r=runtime();a=CacheScope();b=CacheScope();first=request(a)
        r.pending.extend([first,request(a),request(a)])
        with r.condition:
            self.assertIs(r._next(),first)
            newcomer=request(b);r.pending.append(newcomer)
            self.assertIs(r._next(),newcomer)
            r._next();self.assertLessEqual(len(r.fair_admission.recent),1)
        # Identity comparison also supports unhashable internal scope objects.
        f=FairAdmission();one=[];two=[]
        items=[SimpleNamespace(cache_scope=one),SimpleNamespace(cache_scope=two)]
        self.assertIs(f.choose('a',items),items[0]);self.assertIs(f.choose('a',items),items[1])

    def test_model_and_command_barriers_remain_fifo(self):
        r=runtime();a,b=CacheScope(),CacheScope()
        first=request(a);same=request(a);different=request(b,'b');later=request(b)
        r.pending.extend([first,same,different,later])
        with r.condition:
            self.assertIs(r._next(),first);self.assertIs(r._next('a'),same)
            self.assertIsNone(r._next('a'));self.assertIs(r._next(),different)
            self.assertIsNone(r._next('b'));self.assertIs(r._next(),later)
            command=request(None,command={'action':'status'});generation=request(a)
            r.pending.extend([command,generation]);self.assertIsNone(r._next('a'))
            self.assertIs(r._next(),command);self.assertFalse(r.fair_admission.recent)
            self.assertIs(r._next(),generation)

    def test_expired_and_cancelled_nonfront_requests_are_finished_before_turn(self):
        r=runtime();a,b=CacheScope(),CacheScope();live=request(a);expired=request(b);cancelled=request(b);last=request(b)
        expired.start=time.perf_counter()-61;cancelled.cancelled.set()
        r.pending.extend([live,expired,cancelled,last]);r.requests.update(r.pending)
        with r.condition:
            self.assertIs(r._next(),live);self.assertIs(r._next('a'),last)
        self.assertTrue(expired.done.is_set());self.assertIsInstance(expired.error,BusyError)
        self.assertTrue(cancelled.done.is_set());self.assertIsNone(cancelled.error)
        self.assertNotIn(expired,r.requests);self.assertNotIn(cancelled,r.requests)
        self.assertIsNotNone(last.queue_seconds)

    def test_anonymous_single_lane_keeps_fifo(self):
        r=runtime();items=[request(None) for _ in range(6)];r.pending.extend(items)
        with r.condition:self.assertEqual([r._next() for _ in items],items)

    def test_cancellation_between_head_check_and_group_scan(self):
        r=runtime();item=request(CacheScope());r.pending.append(item);r.requests.add(item)
        with patch.object(item.cancelled,'is_set',side_effect=[False,True]):
            with r.condition:self.assertIsNone(r._next())
        self.assertTrue(item.done.is_set());self.assertFalse(r.pending);self.assertFalse(r.fair_admission.recent)

    def test_worker_handoff_preserves_client_turns(self):
        from concurrent.futures import ThreadPoolExecutor
        from damendaemon.request_scope import cache_scope
        from tests.test_scheduler import FakeRuntime
        with patch('damendaemon.scheduler.Runtime',FakeRuntime):
            r=BatchedRuntime({'a':'a'},batch_size=1,queue_size=8)
        executor=r.runtime.executor;order=[];original=executor.prefill
        def prefill(tokens,slot):
            order.append(tokens[0]);return original(tokens,slot)
        executor.prefill=prefill
        def run(scope,prompt):
            token=cache_scope.set(scope)
            try:return list(r.generate('a',[{'content':prompt}],1))
            finally:cache_scope.reset(token)
        a,b,c=CacheScope(),CacheScope(),CacheScope()
        try:
            with ThreadPoolExecutor(5) as pool:
                futures=[pool.submit(run,a,'a0')]
                try:
                    self.assertTrue(executor.entered.wait(2))
                    for count,(scope,prompt) in enumerate(((a,'a1'),(a,'a2'),(b,'b0'),(c,'c0')),1):
                        futures.append(pool.submit(run,scope,prompt))
                        with r.condition:self.assertTrue(r.condition.wait_for(lambda:len(r.pending)==count,timeout=2))
                finally:executor.release.set()
                for future in futures:self.assertEqual(future.result(3)[-1]['usage']['completion_tokens'],1)
            self.assertEqual(order,['a0','b0','c0','a1','a2'])
        finally:executor.release.set();r.close()
