from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.http_auth import AuthPolicy, ClientKey, INFERENCE_ROUTES
from dllm.server import handler_for
from tests.test_server import FakeRuntime

ALICE = 'alice-permissions-test-key'
BOB = 'bob-permissions-test-key-1'
DENIED = 'denied-permissions-test-key'
ADMIN = 'admin-permissions-test-key'


class ClientModelsTests(unittest.TestCase):
    def test_strict_configuration_and_exact_alias_semantics(self):
        for models in ([], ['a'], ['a', 'org/model']):
            raw = json.dumps({'alice': {'key':ALICE, 'max_inflight':1, 'models':models}})
            with patch.dict('os.environ', {'DLLM_CLIENT_KEYS':raw}, clear=True):
                client = AuthPolicy.from_env().clients[0]
            self.assertEqual(client.models, frozenset(models))
            self.assertFalse(client.allows_model('A'))
            self.assertFalse(client.allows_model('a/../b'))
            self.assertNotIn(ALICE, repr(client))
        for models in (None, 'a', ['a','a'], [''], [1], ['a\n'], ['a'*129], [str(i) for i in range(129)]):
            raw = json.dumps({'alice': {'key':ALICE, 'max_inflight':1, 'models':models}})
            with patch.dict('os.environ', {'DLLM_CLIENT_KEYS':raw}, clear=True):
                with self.assertRaises(ValueError) as error:
                    AuthPolicy.from_env()
                self.assertNotIn(ALICE, str(error.exception))
        self.assertTrue(ClientKey('bob', BOB, 1).allows_model('anything'))


class ClientModelsHttpTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.runtime.models = {'a':'unused', 'private':'unused'}
        policy = AuthPolicy(admin_key=ADMIN, clients=(ClientKey('alice', ALICE, 1, frozenset({'a'})),
                            ClientKey('bob', BOB, 1), ClientKey('denied', DENIED, 1, frozenset())))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.runtime, policy))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, key, body=None):
        request = Request(self.url+path, None if body is None else json.dumps(body).encode(),
                          {'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read().decode()

    def test_discovery_is_filtered_but_admin_and_unrestricted_clients_are_unchanged(self):
        for key, expected in ((ALICE, ['a']), (DENIED, []), (BOB, ['a','private']), (ADMIN, ['a','private'])):
            status, body = self.request('/v1/models', key)
            self.assertEqual(status, 200)
            self.assertEqual([m['id'] for m in json.loads(body)['data']], expected)

    def test_every_inference_route_denies_before_runtime_and_releases_lease(self):
        with patch.object(self.runtime, 'generate', side_effect=AssertionError('generation called')), \
                patch.object(self.runtime, 'embed', side_effect=AssertionError('embedding called')), \
                patch.object(self.runtime, 'rerank', side_effect=AssertionError('reranking called')), \
                patch.object(self.runtime, 'score', side_effect=AssertionError('scoring called')):
            for path in INFERENCE_ROUTES:
                for model in ('private', 'unknown', 'A', 'a/../private'):
                    status, body = self.request(path, ALICE, {'model':model, 'stream':True})
                    self.assertEqual(status, 403, (path, model, body))
                    self.assertEqual(json.loads(body), {'error':'forbidden'})
            self.assertEqual(self.request('/v1/completions', DENIED, {'model':'a'})[0], 403)
        self.assertEqual(self.request('/v1/completions', ALICE, {'model':'a', 'prompt':'hello', 'max_tokens':2})[0], 200)
        self.assertEqual(self.request('/v1/completions', ALICE, {'prompt':'hello'})[0], 400)
        status, body = self.request('/v1/completions', ALICE, {'model':'a', 'prompt':'hello', 'max_tokens':2, 'stream':True})
        self.assertEqual(status, 200)
        self.assertIn('data: [DONE]', body)

    def test_allowed_models_work_on_each_endpoint(self):
        payloads = {'/v1/completions':{'prompt':'hello', 'max_tokens':2},
                    '/v1/chat/completions':{'messages':[{'role':'user','content':'hello'}], 'max_tokens':2},
                    '/v1/embeddings':{'input':'hello'},
                    '/v1/rerank':{'query':'hello','documents':['one','two']},
                    '/v1/token-scores':{'prompt':'hello','candidate_token_ids':[1,2]}}
        for path, payload in payloads.items():
            status, body = self.request(path, ALICE, {'model':'a', **payload})
            self.assertEqual(status, 200, (path, body))
