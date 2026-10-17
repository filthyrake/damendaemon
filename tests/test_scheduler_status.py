"""Scheduler-only status stays readable without GPU worker progress."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import io
import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen
from urllib.error import HTTPError

from damendaemon.cli import main
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import handler_for
from tests.test_scheduler import FakeRuntime


class SchedulerStatusTests(unittest.TestCase):
    def setUp(self):
        with patch('damendaemon.scheduler.Runtime', FakeRuntime):
            self.runtime = BatchedRuntime({'a': 'a', 'b': 'b'}, batch_size=2, queue_size=4)
        self.executor = self.runtime.runtime.executor
        self.addCleanup(self.runtime.close)
        self.addCleanup(self.executor.release.set)

    def generate(self, alias):
        return list(self.runtime.generate(alias, [{'content': 'private prompt'}], 2))

    def wait_queued(self, count):
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            with self.runtime.condition:
                if len(self.runtime.pending) == count:
                    return
            time.sleep(.001)
        self.fail('requests did not queue')

    def test_http_and_cli_status_bypass_full_queue_and_blocked_worker(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.runtime))
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with ThreadPoolExecutor(4) as pool:
                futures = [pool.submit(self.generate, 'a')]
                try:
                    self.assertTrue(self.executor.entered.wait(2))
                    futures.append(pool.submit(self.generate, 'b'))
                    self.wait_queued(1)
                    futures.append(pool.submit(self.runtime.manage, 'load', 'a'))
                    self.wait_queued(2)
                    futures.append(pool.submit(self.runtime.manage, 'unload', 'b'))
                    self.wait_queued(3)
                    self.assertFalse(self.runtime.admission.acquire(blocking=False))
                    with urlopen(base+'/v1/scheduler/status', timeout=1) as response:
                        snapshot = json.load(response)
                    self.assertEqual(set(snapshot), {'closed', 'draining', 'drained', 'worker_busy', 'queue_capacity', 'queue_timeout_seconds', 'batch_capacity', 'client_decode_turns', 'client_prefill_turns',
                        'queued_generation_requests', 'worker_owned_generation_requests',
                        'queued_commands', 'worker_owned_commands', 'oldest_queued_seconds',
                        'admission_outstanding', 'admitted_total', 'released_total',
                        'admission_semaphore'})
                    self.assertEqual(snapshot['queue_capacity'], 4)
                    self.assertEqual(snapshot['batch_capacity'], 2)
                    self.assertEqual(snapshot['queued_generation_requests'], 1)
                    self.assertEqual(snapshot['worker_owned_generation_requests'], 1)
                    self.assertEqual(snapshot['queued_commands'], 2)
                    self.assertEqual(snapshot['worker_owned_commands'], 0)
                    self.assertGreater(snapshot['oldest_queued_seconds'], 0)
                    output = io.StringIO()
                    with patch('sys.argv', ['damendaemon', 'status', '--server', base]), patch('sys.stdout', output):
                        main()
                    cli_snapshot = json.loads(output.getvalue())
                    self.assertEqual(cli_snapshot['queued_commands'], 2)
                    self.assertFalse(self.executor.release.is_set())
                    self.assertEqual(self.runtime.runtime.groups, ['a'])
                finally:
                    self.executor.release.set()
                for future in futures:
                    future.result(2)
            snapshot = self.runtime.scheduler_status()
            self.assertEqual(snapshot['queued_generation_requests'], 0)
            self.assertEqual(snapshot['worker_owned_generation_requests'], 0)
            self.assertEqual(snapshot['queued_commands'], 0)
            self.assertEqual(snapshot['worker_owned_commands'], 0)
            self.assertEqual(snapshot['oldest_queued_seconds'], 0)
            self.runtime.close()
            self.assertTrue(self.runtime.scheduler_status()['closed'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_owned_command_and_expired_generation_are_distinct(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_command(**options):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test command not released')
            return {}
        self.runtime.queue_timeout = .05
        with patch.object(self.runtime.runtime, 'manage', side_effect=blocked_command):
            with ThreadPoolExecutor(2) as pool:
                owned = pool.submit(self.runtime.manage, 'load', 'a')
                try:
                    self.assertTrue(entered.wait(1))
                    self.assertEqual(self.runtime.scheduler_status()['worker_owned_commands'], 1)
                    queued = pool.submit(self.generate, 'b')
                    from damendaemon.runtime import BusyError
                    with self.assertRaises(BusyError):
                        queued.result(1)
                    snapshot = self.runtime.scheduler_status()
                    self.assertEqual(snapshot['queued_generation_requests'], 0)
                    self.assertEqual(snapshot['worker_owned_generation_requests'], 0)
                    self.assertEqual(snapshot['worker_owned_commands'], 1)
                finally:
                    release.set()
                owned.result(1)

    def wait_drained(self):
        deadline = time.monotonic()+2
        with self.runtime.condition:
            while not self.runtime.scheduler_status()['drained']:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self.fail('worker did not finish draining')
                self.runtime.condition.wait(remaining)

    def test_http_cli_drain_preserves_admitted_work_and_rejects_new_work(self):
        from urllib.request import Request
        from damendaemon.runtime import BusyError
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.runtime))
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(self.generate, 'a')]
                try:
                    self.assertTrue(self.executor.entered.wait(2))
                    futures.append(pool.submit(self.generate, 'b'))
                    self.wait_queued(1)
                    futures.append(pool.submit(self.runtime.manage, 'load', 'a'))
                    self.wait_queued(2)
                    output = io.StringIO()
                    with patch('sys.argv', ['damendaemon', 'drain', '--server', base]), patch('sys.stdout', output):
                        main()
                    state = json.loads(output.getvalue())
                    self.assertTrue(state['draining'])
                    self.assertFalse(state['drained'])
                    self.assertTrue(state['worker_busy'])
                    self.assertEqual(state['queued_generation_requests'], 1)
                    self.assertEqual(state['queued_commands'], 1)
                    with self.assertRaisesRegex(BusyError, 'draining'):
                        self.generate('a')
                    with self.assertRaisesRegex(BusyError, 'draining'):
                        self.runtime.manage('load', 'a')
                    request = Request(base+'/v1/chat/completions',
                        data=json.dumps({'model': 'a', 'messages': [{'role': 'user', 'content': 'new'}]}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request, timeout=1)
                    self.assertEqual(error.exception.code, 503)
                    error.exception.close()
                    request = Request(base+'/v1/scheduler/drain', data=b'{}',
                                      headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=1) as response:
                        self.assertFalse(json.load(response)['drained'])
                    self.assertFalse(self.executor.release.is_set())
                finally:
                    self.executor.release.set()
                results = [future.result(2) for future in futures]
                self.assertTrue(all(r[-1]['usage']['completion_tokens'] == 2 for r in results[:2]))
                self.assertEqual(results[2], {'done': 'load'})
            self.wait_drained()
            self.assertTrue(self.runtime.drain()['drained'])
            self.assertFalse(self.runtime.scheduler_status()['closed'])
            self.assertEqual(self.runtime.runtime.groups, ['a', 'b', 'control:load'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_drain_waits_for_worker_even_after_caller_is_cancelled(self):
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(self.generate, 'a')
            try:
                self.assertTrue(self.executor.entered.wait(1))
                self.runtime.drain()
                # Simulate the consumer's cancellation cleanup while the worker
                # remains inside prefill; request counts alone now read zero.
                with self.runtime.condition:
                    request = next(iter(self.runtime.requests))
                    request.cancelled.set()
                    self.runtime.requests.discard(request)
                state = self.runtime.scheduler_status()
                self.assertEqual(state['worker_owned_generation_requests'], 0)
                self.assertTrue(state['worker_busy'])
                self.assertFalse(state['drained'])
            finally:
                self.executor.release.set()
            self.assertEqual(future.result(2), [])
        self.wait_drained()

    def test_failed_owned_command_does_not_leave_drain_busy(self):
        entered, release = threading.Event(), threading.Event()
        def fail(**options):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test command not released')
            raise ValueError('command failed')
        with patch.object(self.runtime.runtime, 'manage', side_effect=fail), ThreadPoolExecutor(1) as pool:
            future = pool.submit(self.runtime.manage, 'load', 'a')
            try:
                self.assertTrue(entered.wait(1))
                self.assertFalse(self.runtime.drain()['drained'])
            finally:
                release.set()
            with self.assertRaisesRegex(ValueError, 'command failed'):
                future.result(1)
        self.wait_drained()
        self.assertTrue(self.runtime.drain()['drained'])

    def test_unsupported_runtime_returns_explicit_unavailable_response(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(object()))
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(f'http://127.0.0.1:{server.server_port}/v1/scheduler/status', timeout=1)
            self.assertEqual(error.exception.code, 501)
            self.assertIn('unavailable', json.load(error.exception)['error'])
            error.exception.close()
            from urllib.request import Request
            for body, expected in ((b'{}', 501), (b'{"force":true}', 400)):
                request = Request(f'http://127.0.0.1:{server.server_port}/v1/scheduler/drain',
                                  data=body, headers={'Content-Type': 'application/json'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=1)
                self.assertEqual(error.exception.code, expected)
                error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
