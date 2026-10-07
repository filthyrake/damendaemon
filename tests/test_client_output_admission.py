from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from dllm.client_admission import ClientAdmission,ClientOutputLimitError
from dllm.http_auth import AuthPolicy,ClientKey
from dllm.server import handler_for
from tests.test_server import FakeRuntime

ALICE='alice-output-budget-key-1'
BOB='bob-output-budget-key-123'
ADMIN='admin-output-budget-key-1'


def policy():
    return AuthPolicy(admin_key=ADMIN,clients=(ClientKey('alice',ALICE,4,max_reserved_output_tokens=8),
                                               ClientKey('bob',BOB,4,max_reserved_output_tokens=8)))


class ClientOutputAdmissionTests(unittest.TestCase):
    def test_configuration_exact_boundary_and_exception_release(self):
        for value in (None,True,0,-1,1.5,'8',2**31):
            config={'alice':{'key':ALICE,'max_inflight':4,'max_reserved_output_tokens':value}}
            with patch.dict('os.environ',{'DLLM_CLIENT_KEYS':json.dumps(config)},clear=True):
                with self.assertRaises(ValueError):AuthPolicy.from_env()
        config={'alice':{'key':ALICE,'max_inflight':4,'max_reserved_output_tokens':8}}
        with patch.dict('os.environ',{'DLLM_CLIENT_KEYS':json.dumps(config)},clear=True):
            self.assertEqual(AuthPolicy.from_env().clients[0].max_reserved_output_tokens,8)
        admission=ClientAdmission(policy());a=['Bearer '+ALICE];b=['Bearer '+BOB]
        with admission.output_lease(a,5):
            with admission.output_lease(a,3):
                self.assertEqual(admission.reserved_output['alice'],8)
                with self.assertRaises(ClientOutputLimitError):
                    with admission.output_lease(a,1):pass
                with admission.output_lease(b,8):self.assertEqual(admission.reserved_output['bob'],8)
            self.assertEqual(admission.reserved_output,{'alice':5})
        for failure in (RuntimeError,BrokenPipeError,GeneratorExit):
            with self.assertRaises(failure):
                with admission.output_lease(a,8):raise failure()
            self.assertFalse(admission.reserved_output)
        with admission.output_lease(['Bearer '+ADMIN],100000):self.assertFalse(admission.reserved_output)

    def test_omitted_budget_and_invalid_reservation(self):
        admission=ClientAdmission(AuthPolicy(clients=(ClientKey('alice',ALICE,1),)))
        headers=['Bearer '+ALICE]
        with admission.output_lease(headers,2**40):
            self.assertFalse(admission.reserved_output)
        for value in (True,0,-1,1.5,None):
            with self.assertRaises(ValueError):
                with admission.output_lease(headers,value):pass
        self.assertFalse(admission.reserved_output)

    def test_concurrent_reservations_cannot_overcommit(self):
        admission=ClientAdmission(policy());headers=['Bearer '+ALICE]
        def reserve(_):
            try:
                with admission.output_lease(headers,4):return True
            except ClientOutputLimitError:return False
        with admission.output_lease(headers,5),ThreadPoolExecutor(8) as pool:
            self.assertFalse(any(pool.map(reserve,range(32))))
        self.assertFalse(admission.reserved_output)


class ClientOutputHttpTests(unittest.TestCase):
    def test_both_routes_aliases_concurrent_clients_and_release(self):
        entered,release=threading.Event(),threading.Event()
        calls=[]
        class Runtime(FakeRuntime):
            def generate(self,*args,**kwargs):
                calls.append((args,kwargs))
                if kwargs.get('raw_prompt') == 'fail':raise ValueError('fixture failure')
                if kwargs.get('raw_prompt') == 'hold':
                    entered.set()
                    if not release.wait(5):raise TimeoutError('fixture not released')
                yield from super().generate(*args,**kwargs)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(Runtime(),policy()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def request(key,limit=3,*,route='/v1/completions',field='max_tokens',prompt='Hello',stream=False,model='a'):
            body={'model':model,'stream':stream}
            body.update({'prompt':prompt} if route == '/v1/completions' else {'messages':[{'role':'user','content':prompt}]})
            if field is not None:body[field]=limit
            req=Request(f'http://127.0.0.1:{server.server_port}'+route,json.dumps(body).encode(),
                        {'Content-Type':'application/json','Authorization':'Bearer '+key})
            try:response=urlopen(req,timeout=5)
            except HTTPError as error:response=error
            with response:return response.status,response.getheader('Retry-After'),response.read().decode()
        try:
            with ThreadPoolExecutor(1) as pool:
                held=pool.submit(request,ALICE,6,prompt='hold',stream=True)
                try:
                    self.assertTrue(entered.wait(2))
                    before=len(calls)
                    for route in ('/v1/completions','/v1/chat/completions'):
                        for field in (('max_tokens','max_completion_tokens',None) if route == '/v1/chat/completions' else ('max_tokens',None)):
                            status,retry,body=request(ALICE,3,route=route,field=field)
                            self.assertEqual(status,429,body);self.assertEqual(retry,'1')
                            self.assertNotIn(ALICE,body)
                    self.assertEqual(len(calls),before)
                    self.assertEqual(request(ALICE,2)[0],200)
                    self.assertEqual(request(BOB,8)[0],200)
                    self.assertEqual(request(ADMIN,100)[0],200)
                finally:release.set()
                self.assertEqual(held.result(5)[0],200)
            self.assertEqual(request(ALICE,8,prompt='fail')[0],400)
            self.assertEqual(request(ALICE,8,stream=True)[0],200)
            self.assertEqual(request(ALICE,8,model='unknown')[0],400)
            self.assertEqual(request(ALICE,8)[0],200)
            self.assertEqual(request(ALICE,8,prompt=[])[0],400)
            self.assertEqual(request(ALICE,8)[0],200)
        finally:
            release.set();server.shutdown();server.server_close();thread.join()
