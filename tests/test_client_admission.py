from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from damendaemon.client_admission import ClientAdmission, ClientLimitError
from damendaemon.http_auth import AuthPolicy, ClientKey
from damendaemon.server import handler_for
from tests.test_server import FakeRuntime

ALICE = 'alice-test-key-123456789'
BOB = 'bob-test-key-12345678901'
ADMIN = 'admin-test-key-123456789'


def policy():
    return AuthPolicy(admin_key=ADMIN, clients=(ClientKey('alice', ALICE, 1), ClientKey('bob', BOB, 1)))


class ClientAdmissionTests(unittest.TestCase):
    def test_configuration_and_role_boundaries(self):
        value = json.dumps({'alice': {'key':ALICE, 'max_inflight':2}})
        with patch.dict('os.environ', {'DAMENDAEMON_CLIENT_KEYS':value}, clear=True):
            auth = AuthPolicy.from_env()
        headers = ['Bearer '+ALICE]
        self.assertIsNone(auth.authorize('POST', '/v1/completions', headers))
        self.assertEqual(auth.authorize('POST', '/v1/models/load', headers), 403)
        self.assertEqual(auth.authorize('GET', '/v1/models', []), 401)
        self.assertIsNone(auth.authorize('GET', '/health', []))
        self.assertNotIn(ALICE, repr(auth))
        self.assertNotIn(ALICE, repr(auth.clients[0]))
        invalid = ('', '{}', '[]', '{"alice":{},"alice":{}}',
                   json.dumps({'alice':{'key':ALICE, 'max_inflight':True}}),
                   json.dumps({'alice':{'key':ALICE, 'max_inflight':0}}),
                   json.dumps({'alice':{'key':ALICE, 'max_inflight':129}}),
                   json.dumps({'alice':{'key':ALICE, 'max_inflight':1, 'role':'admin'}}),
                   json.dumps({'alice':{'key':ALICE, 'max_inflight':1},
                               'bob':{'key':ALICE, 'max_inflight':1}}))
        for value in invalid:
            with self.subTest(value=value), patch.dict('os.environ', {'DAMENDAEMON_CLIENT_KEYS':value}, clear=True):
                with self.assertRaises(ValueError) as error:
                    AuthPolicy.from_env()
                self.assertNotIn(ALICE, str(error.exception))
        with self.assertRaises(ValueError):
            AuthPolicy(ALICE, clients=(ClientKey('alice', ALICE, 1),))

    def test_independent_limits_and_release_after_error_or_disconnect(self):
        admission = ClientAdmission(policy())
        a, b = ['Bearer '+ALICE], ['Bearer '+BOB]
        with admission.lease(a):
            with self.assertRaises(ClientLimitError):
                with admission.lease(a):
                    pass
            with admission.lease(b):
                self.assertEqual(admission.active, {'alice':1, 'bob':1})
        self.assertFalse(admission.active)
        for error in (RuntimeError, BrokenPipeError, GeneratorExit):
            with self.assertRaises(error):
                with admission.lease(a):
                    raise error()
            self.assertFalse(admission.active)

    def test_concurrent_attempts_do_not_exceed_a_held_client_limit(self):
        admission = ClientAdmission(policy())
        headers = ['Bearer '+ALICE]
        def attempt(_):
            try:
                with admission.lease(headers):
                    return True
            except ClientLimitError:
                return False
        with admission.lease(headers), ThreadPoolExecutor(max_workers=8) as pool:
            self.assertFalse(any(pool.map(attempt, range(32))))
        self.assertFalse(admission.active)


class ClientAdmissionHttpTests(unittest.TestCase):
    def test_streaming_lease_rejects_before_body_read_and_other_client_proceeds(self):
        entered, release = threading.Event(), threading.Event()
        class BlockingRuntime(FakeRuntime):
            def generate(self, *args, **kwargs):
                if kwargs.get('raw_prompt') == 'hold':
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError('test failed to release request')
                yield from super().generate(*args, **kwargs)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(BlockingRuntime(), policy()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        def call(key, prompt, stream=False):
            payload = dict(model='a', prompt=prompt, max_tokens=2, stream=stream)
            request = Request(url+'/v1/completions', json.dumps(payload).encode(),
                              {'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
            try:
                with urlopen(request, timeout=5) as response:
                    return response.status, response.read().decode()
            except HTTPError as error:
                with error:
                    return error.code, error.read().decode()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(call, ALICE, 'hold', True)
                try:
                    self.assertTrue(entered.wait(2))
                    connection = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                    try:
                        connection.putrequest('POST', '/v1/completions')
                        connection.putheader('Authorization', 'Bearer '+ALICE)
                        connection.putheader('Content-Type', 'application/json')
                        connection.putheader('Content-Length', '100')
                        connection.endheaders()  # Intentionally never send body.
                        response = connection.getresponse()
                        self.assertEqual(response.status, 429)
                        self.assertEqual(response.getheader('Retry-After'), '1')
                        self.assertNotIn(ALICE, response.read().decode())
                    finally:
                        connection.close()
                    self.assertEqual(call(BOB, 'hello')[0], 200)
                finally:
                    release.set()
                status, body = first.result(timeout=5)
                self.assertEqual(status, 200)
                self.assertIn('data: [DONE]', body)
            self.assertEqual(call(ALICE, 'hello')[0], 200)
            self.assertEqual(call(ALICE, [])[0], 400)
            self.assertEqual(call(ALICE, 'hello')[0], 200)
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join()
