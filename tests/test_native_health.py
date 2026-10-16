"""Local lifecycle health does not depend on an idle inference worker."""
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest.mock import patch

from dllm.runtime import Runtime
from dllm.scheduler import BatchedRuntime
from dllm.server import handler_for
from tests.test_scheduler import FakeRuntime


class NativeHealthTests(unittest.TestCase):
    def assert_health(self, runtime, expected):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(runtime))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            try:
                response = urlopen(f'http://127.0.0.1:{server.server_port}/health', timeout=2)
            except HTTPError as error:
                response = error
            with response:
                self.assertEqual(response.status, expected)
                self.assertEqual(json.load(response)['status'], 'ok' if expected == 200 else 'unavailable')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_native_health_ignores_busy_lock_but_observes_shutdown(self):
        runtime = Runtime.__new__(Runtime)
        runtime.closing = runtime.closed = False
        runtime.lock = threading.Lock()
        with runtime.lock:
            self.assert_health(runtime, 200)
            runtime.closing = True
            self.assert_health(runtime, 503)
            runtime.closed = True
            self.assert_health(runtime, 503)

    def test_scheduler_health_tracks_worker_and_admission(self):
        with patch('dllm.scheduler.Runtime', FakeRuntime):
            runtime = BatchedRuntime({'a': 'a'})
        try:
            self.assert_health(runtime, 200)
            with patch.object(runtime.worker, 'is_alive', return_value=False):
                self.assert_health(runtime, 503)
            runtime.drain()
            self.assert_health(runtime, 503)
            runtime.close()
            self.assert_health(runtime, 503)
        finally:
            runtime.close()
