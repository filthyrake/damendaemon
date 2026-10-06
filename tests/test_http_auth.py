"""Real loopback authentication tests; runtime work stays CPU-only."""
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request

from dllm.http_auth import AuthPolicy, authenticated_urlopen
from dllm.server import handler_for
from dllm.cli import main
from dllm.bench import request_one as benchmark_request
from tests.test_server import FakeRuntime

API_KEY = 'inference-test-key-123456'
ADMIN_KEY = 'admin-test-key-123456789'


class HttpAuthTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.runtime.manage = Mock(wraps=self.runtime.manage)
        self.runtime.scheduler_status = Mock(return_value={'drained': False})
        self.runtime.drain = Mock(return_value={'draining': True})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0),
            handler_for(self.runtime, AuthPolicy(API_KEY, ADMIN_KEY)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def call(self, path, key=None, body=None):
        headers = {'Content-Type': 'application/json'}
        if key is not None:
            headers['Authorization'] = 'Bearer '+key
        request = Request(self.url+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with authenticated_urlopen(request, timeout=2) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_roles_and_denied_mutations(self):
        self.assertEqual(self.call('/health')[0], 200)
        self.assertEqual(self.call('/v1/models')[0], 401)
        self.assertEqual(self.call('/v1/models', API_KEY)[0], 200)
        body = {'model': 'a', 'prompt': 'hello', 'max_tokens': 1}
        self.assertEqual(self.call('/v1/completions', API_KEY, body)[0], 200)
        self.assertEqual(self.call('/v1/models/remove', API_KEY, {'model':'a'})[0], 403)
        self.assertEqual(self.call('/v1/models/config', API_KEY)[0], 403)
        self.assertEqual(self.call('/v1/scheduler/drain', API_KEY, {})[0], 403)
        self.runtime.manage.assert_not_called()
        self.runtime.drain.assert_not_called()
        self.assertEqual(self.call('/v1/models/load', ADMIN_KEY, {'model':'a'})[0], 200)
        self.runtime.manage.assert_called_once()
        self.assertEqual(self.call('/v1/scheduler/status', ADMIN_KEY)[0], 200)
        self.assertEqual(self.call('/v1/scheduler/drain', ADMIN_KEY, {})[0], 200)

    def test_rejects_before_waiting_for_declared_request_body(self):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            connection.putrequest('POST', '/v1/models/load')
            connection.putheader('Content-Type', 'application/json')
            connection.putheader('Content-Length', '1048576')
            connection.endheaders()  # Deliberately never send the body.
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            self.assertEqual(response.getheader('WWW-Authenticate'), 'Bearer realm="dllm"')
            response.read()
            self.runtime.manage.assert_not_called()
        finally:
            connection.close()

    def test_duplicate_headers_rejected(self):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            connection.putrequest('GET', '/v1/models')
            connection.putheader('Authorization', 'Bearer '+API_KEY)
            connection.putheader('Authorization', 'Bearer '+ADMIN_KEY)
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
        finally:
            connection.close()

    def test_cli_admin_and_benchmark_inference_credentials(self):
        output = io.StringIO()
        with patch.dict('os.environ', DLLM_API_KEY=API_KEY, DLLM_ADMIN_API_KEY=ADMIN_KEY), \
                patch('sys.argv', ['dllm', 'status', '--server', self.url]), patch('sys.stdout', output):
            main()
        self.assertEqual(json.loads(output.getvalue()), {'drained':False})
        result = benchmark_request(self.url+'/v1', 'a', 'hello', 1, 2, 0, API_KEY)
        self.assertIsNone(result.error)
        self.assertNotIn(API_KEY, repr(result))
        self.assertNotIn(ADMIN_KEY, output.getvalue())


class AuthPolicyTests(unittest.TestCase):
    def test_invalid_configuration_and_headers_never_echo_credentials(self):
        for value in ('', 'short', 'x'*513, 'x'*16+'\n', 'é'*16):
            with self.assertRaises(ValueError):
                AuthPolicy(value)
        with self.assertRaises(ValueError):
            AuthPolicy(API_KEY, API_KEY)
        policy = AuthPolicy(API_KEY, ADMIN_KEY)
        self.assertNotIn(API_KEY, repr(policy))
        self.assertNotIn(ADMIN_KEY, repr(policy))
        for headers in ([], ['Basic '+API_KEY], ['Bearer '+'é'*20], ['Bearer bad'],
                        ['Bearer '+API_KEY, 'Bearer '+API_KEY]):
            self.assertEqual(policy.authorize('GET', '/v1/models', headers), 401)
        self.assertEqual(AuthPolicy(API_KEY).authorize('POST', '/v1/scheduler/drain', ['Bearer '+API_KEY]), 403)
        self.assertIsNone(AuthPolicy(admin_key=ADMIN_KEY).authorize('POST', '/v1/completions', ['Bearer '+ADMIN_KEY]))
        self.assertIsNone(AuthPolicy().authorize('POST', '/v1/models/remove', []))

    def test_bad_server_key_rejects_before_model_resolution(self):
        errors = io.StringIO()
        with patch.dict('os.environ', {'DLLM_API_KEY': 'short'}, clear=True), \
                patch('sys.argv', ['dllm', 'serve', '--model', 'a=Qwen/not-downloaded']), \
                patch('sys.stderr', errors), patch('dllm.cli.resolve_model') as resolve:
            with self.assertRaises(SystemExit):
                main()
            resolve.assert_not_called()
        self.assertIn('DLLM_API_KEY', errors.getvalue())
        self.assertNotIn('short', errors.getvalue())

    def test_authenticated_redirect_does_not_reach_second_endpoint(self):
        calls = []
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                calls.append(self.path)
                self.send_response(302 if self.path == '/start' else 200)
                self.send_header('Location', '/destination')
                self.send_header('Content-Length', '0')
                self.end_headers()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/start'
            with self.assertRaises(HTTPError) as result:
                authenticated_urlopen(Request(url, headers={'Authorization':'Bearer '+API_KEY}), timeout=2)
            result.exception.close()
            self.assertEqual(calls, ['/start'])
            with authenticated_urlopen(url, timeout=2) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(calls, ['/start', '/start', '/destination'])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
