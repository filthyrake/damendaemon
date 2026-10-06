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

from dllm.cli import main
from dllm.scheduler import BatchedRuntime
from dllm.server import handler_for
from tests.test_scheduler import FakeRuntime


class SchedulerStatusTests(unittest.TestCase):
    def setUp(self):
        with patch('dllm.scheduler.Runtime', FakeRuntime):
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
                    self.assertEqual(set(snapshot), {'closed', 'queue_capacity', 'batch_capacity',
                        'queued_generation_requests', 'worker_owned_generation_requests',
                        'queued_commands', 'worker_owned_commands', 'oldest_queued_seconds'})
                    self.assertEqual(snapshot['queue_capacity'], 4)
                    self.assertEqual(snapshot['batch_capacity'], 2)
                    self.assertEqual(snapshot['queued_generation_requests'], 1)
                    self.assertEqual(snapshot['worker_owned_generation_requests'], 1)
                    self.assertEqual(snapshot['queued_commands'], 2)
                    self.assertEqual(snapshot['worker_owned_commands'], 0)
                    self.assertGreater(snapshot['oldest_queued_seconds'], 0)
                    output = io.StringIO()
                    with patch('sys.argv', ['dllm', 'status', '--server', base]), patch('sys.stdout', output):
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
                    from dllm.runtime import BusyError
                    with self.assertRaises(BusyError):
                        queued.result(1)
                    snapshot = self.runtime.scheduler_status()
                    self.assertEqual(snapshot['queued_generation_requests'], 0)
                    self.assertEqual(snapshot['worker_owned_generation_requests'], 0)
                    self.assertEqual(snapshot['worker_owned_commands'], 1)
                finally:
                    release.set()
                owned.result(1)

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
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
