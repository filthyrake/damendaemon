from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from dllm.distributed_serving import DistributedRuntime
from dllm.runtime import BusyError
from dllm.http_auth import AuthPolicy
from tests.test_distributed_serving import Pair,RuntimeHelpers
from tests import test_host_serving as fixtures


def wait_for(predicate):
    deadline=time.monotonic()+3
    while not predicate() and time.monotonic()<deadline:time.sleep(.001)
    if not predicate():raise AssertionError('condition did not become true')


class DistributedDrainTests(RuntimeHelpers,unittest.TestCase):
    def test_drain_and_close_racing_preparation_never_enqueue(self):
        for action in ('drain','close'):
            pair=Pair();runtime=DistributedRuntime(pair);entered=threading.Event();resume=threading.Event()
            def encode(*args,**kwargs):
                entered.set();self.assertTrue(resume.wait(3));return [1]
            with patch('dllm.distributed_serving.encode',side_effect=encode),ThreadPoolExecutor(1) as pool:
                future=pool.submit(lambda:list(runtime.generate('model',None,1,raw_prompt=[1])))
                self.assertTrue(entered.wait(3))
                getattr(runtime,action)();resume.set()
                with self.assertRaises(BusyError):future.result(3)
            self.assertTrue(runtime.requests.empty());self.assertEqual(runtime.accepted,{})
            self.assertEqual(pair.calls,[]);runtime.close()

    def test_close_flushes_accepted_queue_without_started_owner(self):
        runtime=DistributedRuntime(Pair())
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(lambda:list(runtime.generate('model',None,1,raw_prompt=[1])))
            wait_for(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            runtime.close()
            with self.assertRaises(BusyError):future.result(3)
        self.assertTrue(runtime.requests.empty());self.assertEqual(runtime.accepted,{})
        self.assertFalse(runtime.drain()['drained'])


class CpuPeerDrainTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.HostServingTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.runtime=self.fixture.runtime

    def test_queued_work_finishes_then_admin_status_reports_drained(self):
        inference,admin=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        base=self.fixture.serve(AuthPolicy(inference,admin))
        def drain(key):
            return urlopen(Request(base+'/v1/scheduler/drain',b'{}',{'Content-Type':'application/json',
                'Authorization':'Bearer '+key}),timeout=3)
        with self.assertRaises(HTTPError) as error:drain(inference)
        self.assertEqual(error.exception.code,403)
        with ThreadPoolExecutor(3) as pool:
            futures=[pool.submit(lambda:list(self.runtime.generate('model',None,3,raw_prompt=[1,2]))) for _ in range(3)]
            wait_for(lambda:self.runtime.scheduler_status()['queued_generation_requests']==3)
            with drain(admin) as response:state=json.load(response)
            self.assertTrue(state['draining']);self.assertFalse(state['drained'])
            self.assertEqual(state['queued_generation_requests'],3)
            self.fixture.start(heartbeat_interval=.05)
            results=[future.result(5) for future in futures]
        self.assertEqual([r[-1]['usage']['completion_tokens'] for r in results],[3,3,3])
        wait_for(lambda:self.runtime.scheduler_status()['drained'])
        self.assertFalse(any(self.fixture.pair.state.lengths));self.assertFalse(self.runtime.closed.is_set())
        with urlopen(Request(base+'/v1/scheduler/status',headers={'Authorization':'Bearer '+admin}),timeout=3) as response:
            self.assertTrue(json.load(response)['drained'])
        with drain(admin) as response:self.assertTrue(json.load(response)['drained'])
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base+'/v1/completions',json.dumps(dict(model='model',prompt=[1],max_tokens=1)).encode(),
                {'Content-Type':'application/json','Authorization':'Bearer '+inference}),timeout=3)
        self.assertEqual(error.exception.code,503)
        before=self.runtime.heartbeats;wait_for(lambda:self.runtime.heartbeats>before)
        self.assertFalse(self.fixture.errors)

    def test_cancelled_backpressure_requires_release_ack_before_drained(self):
        self.fixture.start(heartbeat_interval=.05)
        stream=self.runtime.generate('model',None,14,raw_prompt=[1])
        next(stream)
        wait_for(lambda:self.runtime.requests.empty() and self.runtime.active)
        self.assertFalse(self.runtime.drain()['drained'])
        entered=threading.Event();resume=threading.Event();release=self.fixture.pair.release
        def held_release(*args):
            entered.set();self.assertTrue(resume.wait(3));return release(*args)
        with patch.object(self.fixture.pair,'release',side_effect=held_release):
            stream.close();self.assertTrue(entered.wait(3))
            self.assertFalse(self.runtime.scheduler_status()['drained'])
            self.assertEqual(self.runtime.scheduler_status()['worker_owned_generation_requests'],1)
            resume.set();wait_for(lambda:self.runtime.scheduler_status()['drained'])
        self.assertFalse(any(self.fixture.pair.state.lengths));self.assertFalse(self.fixture.errors)

    def test_release_failure_cannot_report_successful_drain(self):
        with patch.object(self.fixture.pair,'release',side_effect=RuntimeError('release failed')):
            with ThreadPoolExecutor(1) as pool:
                result=pool.submit(lambda:list(self.runtime.generate('model',None,1,raw_prompt=[1])))
                wait_for(lambda:self.runtime.scheduler_status()['queued_generation_requests']==1)
                self.runtime.drain();self.fixture.start()
                with self.assertRaises(RuntimeError):result.result(5)
                self.fixture.owner.join(3)
        state=self.runtime.scheduler_status()
        self.assertTrue(state['failed']);self.assertTrue(state['closed']);self.assertFalse(state['drained'])
        self.assertTrue(self.fixture.errors)
