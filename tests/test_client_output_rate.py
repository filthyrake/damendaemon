from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from damendaemon.client_admission import ClientAdmission, ClientOutputLimitError, ClientOutputRateLimitError
from damendaemon.http_auth import AuthPolicy, ClientKey
from damendaemon.server import handler_for
from tests.test_server import FakeRuntime

ALICE = 'alice-output-rate-key-1234'
BOB = 'bob-output-rate-key-123456'
ADMIN = 'admin-output-rate-key-1234'


def policy(limit=10, cap=None):
    return AuthPolicy(admin_key=ADMIN, clients=tuple(
        ClientKey(name, key, 128, max_reserved_output_tokens=cap,
                  max_output_tokens_per_minute=limit)
        for name, key in (('alice', ALICE), ('bob', BOB))))


class OutputRateTests(unittest.TestCase):
    def test_configuration_and_unlimited_default(self):
        for value in (None, True, 0, -1, 1.5, '1', 2**31):
            raw={'alice':dict(key=ALICE, max_inflight=1, max_output_tokens_per_minute=value)}
            with patch.dict('os.environ', {'DAMENDAEMON_CLIENT_KEYS':json.dumps(raw)}, clear=True):
                with self.assertRaises(ValueError):AuthPolicy.from_env()
        raw['alice']['max_output_tokens_per_minute']=2**31-1
        with patch.dict('os.environ', {'DAMENDAEMON_CLIENT_KEYS':json.dumps(raw)}, clear=True):
            self.assertEqual(AuthPolicy.from_env().clients[0].max_output_tokens_per_minute, 2**31-1)
        admission=ClientAdmission(AuthPolicy(clients=(ClientKey('alice', ALICE, 1),)))
        with admission.output_lease(['Bearer '+ALICE], 2**40):pass
        self.assertFalse(admission.output_windows)
        self.assertFalse(admission.reserved_output)

    def test_weighted_retry_waits_for_enough_budget_and_conservative_expiry(self):
        now=[.1];admission=ClientAdmission(policy(), clock=lambda:now[0]);headers=['Bearer '+ALICE]
        with admission.output_lease(headers, 2):pass
        now[0]=10.2
        with admission.output_lease(headers, 8):pass
        now[0]=60.9
        with self.assertRaises(ClientOutputRateLimitError) as caught:
            with admission.output_lease(headers, 3):pass
        self.assertEqual(caught.exception.retry_after, 11)  # Two expiring tokens are insufficient.
        with self.assertRaises(ClientOutputRateLimitError):
            with admission.output_lease(headers, 2):pass
        now[0]=61
        with admission.output_lease(headers, 2):pass
        self.assertEqual(list(admission.output_windows['alice']), [(71, 8), (122, 2)])
        with self.assertRaises(ValueError):
            with admission.output_lease(headers, 11):pass
        now[0]=71
        with admission.output_lease(headers, 8):pass

    def test_rejections_do_not_charge_but_execution_failures_do(self):
        admission=ClientAdmission(policy(cap=5), clock=lambda:0);headers=['Bearer '+ALICE]
        with admission.output_lease(headers, 5):
            with self.assertRaises(ClientOutputLimitError):
                with admission.output_lease(headers, 1):pass
            self.assertEqual(list(admission.output_windows['alice']), [(61,5)])
        with self.assertRaises(GeneratorExit):
            with admission.output_lease(headers, 5):raise GeneratorExit()
        self.assertFalse(admission.reserved_output)
        with self.assertRaises(ClientOutputRateLimitError):
            with admission.output_lease(headers, 1):pass
        self.assertFalse(admission.reserved_output)
        with admission.output_lease(['Bearer '+BOB], 5):pass
        with admission.output_lease(['Bearer '+ADMIN], 100000):pass
        self.assertEqual(set(admission.output_windows), {'alice','bob'})

    def test_concurrent_charge_and_bucket_storage_bounds(self):
        admission=ClientAdmission(policy(), clock=lambda:.5)
        def attempt(_):
            try:
                with admission.output_lease(['Bearer '+ALICE], 3):return True
            except ClientOutputRateLimitError:return False
        with ThreadPoolExecutor(16) as pool:self.assertEqual(sum(pool.map(attempt, range(100))), 3)
        self.assertEqual(list(admission.output_windows['alice']), [(61,9)])
        now=[0.];admission=ClientAdmission(policy(limit=1000000), clock=lambda:now[0])
        for step in range(1500):
            now[0]=step / 10
            with admission.output_lease(['Bearer '+ALICE], 1):pass
            self.assertLessEqual(len(admission.output_windows['alice']), 61)
        self.assertLessEqual(sum(count for _,count in admission.output_windows['alice']), 610)

    def test_http_aliases_streaming_errors_and_expiry_before_dispatch(self):
        now=[0.];admission=ClientAdmission(policy(), clock=lambda:now[0]);calls=[]
        class Runtime(FakeRuntime):
            def generate(self, *args, **kwargs):
                calls.append(args[2])
                if kwargs.get('raw_prompt')=='fail':raise ValueError('fixture failure')
                yield from super().generate(*args, **kwargs)
        with patch('damendaemon.client_admission.ClientAdmission', return_value=admission):
            handler=handler_for(Runtime(), policy())
        server=ThreadingHTTPServer(('127.0.0.1',0), handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def request(tokens, *, chat=False, stream=False, prompt='Hello', key=ALICE):
            body=dict(model='a',stream=stream)
            if chat:body.update(messages=[dict(role='user',content=prompt)],max_completion_tokens=tokens)
            else:body.update(prompt=prompt,max_tokens=tokens)
            req=Request(f'http://127.0.0.1:{server.server_port}/v1/'+('chat/completions' if chat else 'completions'),
                        json.dumps(body).encode(), {'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            try:response=urlopen(req,timeout=3)
            except HTTPError as error:response=error
            with response:return response.status,response.getheader('Retry-After'),response.read().decode()
        try:
            self.assertEqual(request(11)[0],400)
            self.assertFalse(calls)
            self.assertEqual(request(3,prompt=[])[0],400)
            self.assertFalse(admission.output_windows)
            self.assertEqual(request(4,prompt='fail')[0],400)
            self.assertEqual(request(6,chat=True,stream=True)[0],200)
            before=len(calls)
            for chat in (False,True):
                status,retry,body=request(1,chat=chat,stream=True)
                self.assertEqual(status,429,body);self.assertEqual(retry,'61');self.assertNotIn(ALICE,body)
            self.assertEqual(len(calls),before)
            self.assertEqual(request(10,key=BOB)[0],200)
            self.assertEqual(request(100,key=ADMIN)[0],200)
            now[0]=61
            self.assertEqual(request(10)[0],200)
        finally:server.shutdown();server.server_close();thread.join()
        # Reading the response can precede the handler's outer admission exit.
        # server_close joins request threads, making this a cleanup assertion.
        self.assertFalse(admission.active);self.assertFalse(admission.reserved_output)
