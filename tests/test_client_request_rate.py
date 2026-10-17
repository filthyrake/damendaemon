from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from damendaemon.client_admission import ClientAdmission,ClientLimitError,ClientRateLimitError
from damendaemon.http_auth import AuthPolicy,ClientKey
from damendaemon.server import handler_for
from tests.test_server import FakeRuntime

ALICE='alice-rate-limit-key-1234'
BOB='bob-rate-limit-key-123456'
ADMIN='admin-rate-limit-key-1234'


def policy(limit=2,inflight=4):
    return AuthPolicy(admin_key=ADMIN,clients=(ClientKey('alice',ALICE,inflight,max_requests_per_minute=limit),
                                              ClientKey('bob',BOB,inflight,max_requests_per_minute=limit)))


class ClientRequestRateTests(unittest.TestCase):
    def test_strict_configuration_and_omitted_unlimited(self):
        for value in (None,True,0,-1,1.5,'2',60001):
            with patch.dict('os.environ',{'DAMENDAEMON_CLIENT_KEYS':json.dumps({'alice':dict(key=ALICE,max_inflight=1,max_requests_per_minute=value)})},clear=True):
                with self.assertRaises(ValueError):AuthPolicy.from_env()
        with patch.dict('os.environ',{'DAMENDAEMON_CLIENT_KEYS':json.dumps({'alice':dict(key=ALICE,max_inflight=1,max_requests_per_minute=2)})},clear=True):
            self.assertEqual(AuthPolicy.from_env().clients[0].max_requests_per_minute,2)
        admission=ClientAdmission(AuthPolicy(clients=(ClientKey('alice',ALICE,1),)))
        for _ in range(5):
            with admission.lease(['Bearer '+ALICE]):pass
        self.assertFalse(admission.requests)

    def test_exact_rolling_window_retry_client_isolation_and_failure_charges(self):
        now=[0.];admission=ClientAdmission(policy(),clock=lambda:now[0]);headers=['Bearer '+ALICE]
        with admission.lease(headers):pass
        now[0]=.25
        with self.assertRaises(RuntimeError):
            with admission.lease(headers):raise RuntimeError('request failed')
        now[0]=59.25
        with self.assertRaises(ClientRateLimitError) as error:
            with admission.lease(headers):pass
        self.assertEqual(error.exception.retry_after,1)
        with admission.lease(['Bearer '+BOB]):pass
        for _ in range(3):
            with admission.lease(['Bearer '+ADMIN]):pass
        now[0]=60.
        with admission.lease(headers):pass
        self.assertEqual(list(admission.requests['alice']),[.25,60.])
        with self.assertRaises(ClientRateLimitError):
            with admission.lease(headers):pass
        now[0]=60.25
        with admission.lease(headers):pass
        self.assertFalse(admission.active)
        self.assertEqual(list(admission.requests['alice']),[60.,60.25])

    def test_outstanding_rejection_does_not_consume_rate_and_close_does_not_refund(self):
        admission=ClientAdmission(policy(inflight=1),clock=lambda:0.)
        headers=['Bearer '+ALICE]
        with admission.lease(headers):
            with self.assertRaises(ClientLimitError):
                with admission.lease(headers):pass
            self.assertEqual(len(admission.requests['alice']),1)
        with self.assertRaises(GeneratorExit):
            with admission.lease(headers):raise GeneratorExit()
        self.assertEqual(len(admission.requests['alice']),2);self.assertFalse(admission.active)

    def test_concurrent_attempts_never_over_admit_or_grow_past_limit(self):
        admission=ClientAdmission(policy(limit=3,inflight=128),clock=lambda:0.)
        def attempt(_):
            try:
                with admission.lease(['Bearer '+ALICE]):return True
            except ClientRateLimitError:return False
        with ThreadPoolExecutor(16) as pool:self.assertEqual(sum(pool.map(attempt,range(100))),3)
        self.assertEqual(len(admission.requests['alice']),3);self.assertFalse(admission.active)

    def test_http_shared_across_routes_before_runtime_and_resets_at_expiry(self):
        now=[0.];auth=policy();admission=ClientAdmission(auth,clock=lambda:now[0]);calls=[]
        class Runtime(FakeRuntime):
            def generate(self,*args,**kwargs):calls.append('generate');yield from super().generate(*args,**kwargs)
        with patch('damendaemon.client_admission.ClientAdmission',return_value=admission):handler=handler_for(Runtime(),auth)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def request(key,route='/v1/completions',body=None,method='POST'):
            if body is None:body=dict(model='a',prompt='Hi',max_tokens=2)
            req=Request(f'http://127.0.0.1:{server.server_port}'+route,
                        None if method=='GET' else json.dumps(body).encode(),
                        {'Authorization':'Bearer '+key,'Content-Type':'application/json'},method=method)
            try:response=urlopen(req,timeout=3)
            except HTTPError as error:response=error
            with response:return response.status,response.getheader('Retry-After'),response.read().decode()
        try:
            self.assertEqual(request(ALICE)[0],200)
            self.assertEqual(request(ALICE,body=dict(model='a',prompt=[]))[0],400)
            count=len(calls)
            for route in ('/v1/completions','/v1/chat/completions','/v1/embeddings','/v1/token-scores','/v1/rerank'):
                status,retry,body=request(ALICE,route)
                self.assertEqual(status,429,body);self.assertEqual(retry,'60');self.assertNotIn(ALICE,body)
            self.assertEqual(len(calls),count)
            self.assertEqual(request(ALICE,'/v1/models',method='GET')[0],200)
            self.assertEqual(request(BOB)[0],200)
            for _ in range(3):self.assertEqual(request(ADMIN)[0],200)
            now[0]=60.
            self.assertEqual(request(ALICE,'/v1/chat/completions',dict(model='a',messages=[dict(role='user',content='Hi')],max_tokens=2,stream=True))[0],200)
        finally:server.shutdown();server.server_close();thread.join()
