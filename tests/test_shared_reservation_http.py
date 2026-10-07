from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
import secrets
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.http_auth import AuthPolicy,ClientKey
from dllm.client_admission import ClientAdmission,ClientRateLimitError,ClientOutputRateLimitError
from dllm.shared_client_reservations import SharedClientReservations,SharedReservationUnavailable
from dllm.shared_client_rates import SharedClientRates
from dllm.server import BoundedHTTPServer,handler_for
from tests.test_server import FakeRuntime
from tests.test_distributed_drain import wait_for


class SharedReservationHttpTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)/'claims.db'
        self.client=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=6)
        self.other=ClientKey('bob',secrets.token_urlsafe(32),1,max_reserved_output_tokens=3)
        self.admin=secrets.token_urlsafe(32);self.auth=AuthPolicy(admin_key=self.admin,clients=(self.client,self.other))
        self.servers=[];self.events=[]
        def cleanup():
            for event in self.events:event.set()
            for server,thread in reversed(self.servers):
                server.shutdown();thread.join(3)
                admission=server.RequestHandlerClass.client_admission
                wait_for(lambda:not admission.active and not admission.reserved_output)
                admission.retry_releases();server.server_close()
        self.addCleanup(cleanup)

    def serve(self,runtime=None):
        with patch.dict('os.environ',{'DLLM_CLIENT_RESERVATION_DB':str(self.path)},clear=True):
            handler=handler_for(runtime or FakeRuntime(),self.auth)
        server=BoundedHTTPServer(('127.0.0.1',0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.servers.append((server,thread))
        return server

    def call(self,server,key=None,**options):
        payload=dict(model='a',prompt='hello',max_tokens=2);payload.update(options)
        request=Request(f'http://127.0.0.1:{server.server_port}/v1/completions',json.dumps(payload).encode(),
            {'Authorization':'Bearer '+(key or self.client.key),'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=5) as response:return response.status,response.read().decode()
        except HTTPError as error:
            with error:return error.code,error.read().decode()

    def blocker(self):
        entered,release=threading.Event(),threading.Event();self.events.append(release)
        class BlockingRuntime(FakeRuntime):
            def generate(self,*args,**kwargs):
                if kwargs.get('raw_prompt')=='hold':
                    entered.set()
                    if not release.wait(5):raise TimeoutError('test release missing')
                yield from super().generate(*args,**kwargs)
        return BlockingRuntime(),entered,release

    def test_two_http_servers_share_request_and_output_caps_with_client_isolation(self):
        runtime,entered,release=self.blocker();a,b=self.serve(runtime),self.serve()
        with ThreadPoolExecutor(1) as pool:
            first=pool.submit(self.call,a,prompt='hold',max_tokens=5,stream=True)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.call(b,max_tokens=2)[0],429)
            self.assertEqual(self.call(b,self.other.key)[0],200)
            # A separate non-output lease exhausts the shared request count.
            admission=b.RequestHandlerClass.client_admission
            with admission.lease(['Bearer '+self.client.key]):
                connection=HTTPConnection('127.0.0.1',b.server_port,timeout=2)
                try:
                    connection.putrequest('POST','/v1/completions');connection.putheader('Authorization','Bearer '+self.client.key)
                    connection.putheader('Content-Type','application/json');connection.putheader('Content-Length','100');connection.endheaders()
                    response=connection.getresponse();self.assertEqual(response.status,429);response.read()
                finally:connection.close()
            self.assertEqual(self.call(b,self.admin)[0],200)
            release.set();status,body=first.result(5);self.assertEqual(status,200);self.assertIn('[DONE]',body)
        wait_for(lambda:not a.RequestHandlerClass.client_admission.active)
        self.assertEqual(self.call(b,max_tokens=6)[0],200)
        self.assertEqual(self.call(a,prompt=[])[0],400)
        self.assertEqual(self.call(b)[0],200)

    def test_shared_rates_remain_charged_after_failed_requests_and_release(self):
        client=ClientKey('alice',self.client.key,2,max_reserved_output_tokens=6,max_requests_per_minute=1,max_output_tokens_per_minute=3)
        auth=AuthPolicy(clients=(client,));headers=['Bearer '+client.key]
        a=ClientAdmission(auth,reservation_store=SharedClientReservations(self.path,[client]),rate_store=SharedClientRates(self.path.with_suffix('.rates'),[client]))
        self.addCleanup(a.close)
        with self.assertRaises(RuntimeError):
            with a.lease(headers),a.output_lease(headers,3):raise RuntimeError('model failed')
        self.assertFalse(a.reservation_store.claims)
        with self.assertRaises(ClientRateLimitError):
            with a.lease(headers):pass
        with self.assertRaises(ClientOutputRateLimitError):
            with a.output_lease(headers,1):pass
        self.assertFalse(a.reservation_store.claims)

    def test_storage_error_is_redacted_503_and_failed_reserve_is_retryable(self):
        a=self.serve();admission=a.RequestHandlerClass.client_admission;store=admission.reservation_store
        with patch.object(store,'_commit',side_effect=sqlite3.OperationalError(self.client.key)):
            status,body=self.call(a)
        self.assertEqual(status,503);self.assertNotIn(self.client.key,body);self.assertEqual(len(admission.pending_releases),1)
        self.assertEqual(self.call(a)[0],503)
        admission.retry_releases();self.assertFalse(store.claims);self.assertEqual(self.call(a)[0],200)

    def test_failed_output_release_suppresses_done_and_retains_capacity(self):
        a,b=self.serve(),self.serve();admission=a.RequestHandlerClass.client_admission;store=admission.reservation_store
        release=store.release
        def fail_output(claim):
            if claim.requests==0:raise SharedReservationUnavailable(store)
            release(claim)
        with patch.object(store,'release',fail_output):status,body=self.call(a,max_tokens=6,stream=True)
        self.assertEqual(status,200);self.assertIn('reservation accounting unavailable',body);self.assertNotIn('[DONE]',body)
        wait_for(lambda:not admission.active)
        self.assertEqual(self.call(b)[0],429);self.assertTrue(admission.pending_releases)
        admission.retry_releases();self.assertEqual(self.call(b)[0],200)

    def test_failed_request_release_never_writes_second_http_response(self):
        a=self.serve();admission=a.RequestHandlerClass.client_admission;store=admission.reservation_store;release=store.release
        def fail_request(claim):
            if claim.requests:raise SharedReservationUnavailable(store)
            release(claim)
        connection=socket.create_connection(('127.0.0.1',a.server_port),timeout=3)
        body=json.dumps(dict(model='a',prompt='hello',max_tokens=2)).encode()
        wire=(f'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {self.client.key}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()+body
        try:
            with patch.object(store,'release',fail_request):
                connection.sendall(wire);chunks=[]
                while True:
                    data=connection.recv(65536)
                    if not data:break
                    chunks.append(data)
            response=b''.join(chunks);self.assertEqual(response.count(b'HTTP/1.1'),1);self.assertIn(b'200 OK',response)
        finally:connection.close()
        self.assertTrue(admission.pending_releases);admission.retry_releases()

    def test_close_refuses_live_leases_and_retries_ended_failed_release(self):
        with patch.dict('os.environ',{},clear=True),patch('dllm.shared_client_reservations.fcntl',None):
            local=ClientAdmission(self.auth)
            with self.assertRaisesRegex(ValueError,'POSIX'):SharedClientReservations(self.path,self.auth.clients)
        self.assertIsNone(local.reservation_store);self.assertFalse(self.path.exists());local.close()
        store=SharedClientReservations(self.path,self.auth.clients);a=ClientAdmission(self.auth,reservation_store=store)
        headers=['Bearer '+self.client.key]
        try:
            with a.lease(headers):
                with self.assertRaises(RuntimeError):a.close()
            release=store.release
            with patch.object(store,'release',side_effect=SharedReservationUnavailable(store)):
                with self.assertRaises(SharedReservationUnavailable):
                    with a.lease(headers):pass
                with self.assertRaises(SharedReservationUnavailable):a.close()
            self.assertIsNotNone(store.descriptor);self.assertFalse(a.closed);a.close();self.assertTrue(store.closed)
        finally:
            a.retry_releases();a.close()

    def test_server_bind_failure_releases_new_owner_and_close_preserves_active_claim(self):
        a=self.serve()
        with patch.dict('os.environ',{'DLLM_CLIENT_RESERVATION_DB':str(self.path)},clear=True):handler=handler_for(FakeRuntime(),self.auth)
        with self.assertRaises(OSError):BoundedHTTPServer(('127.0.0.1',a.server_port),handler)
        self.assertTrue(handler.client_admission.reservation_store.closed)
        admission=a.RequestHandlerClass.client_admission
        with admission.lease(['Bearer '+self.client.key]):
            with self.assertRaises(RuntimeError):a.server_close()
            self.assertIsNotNone(admission.reservation_store.descriptor)
        a.server_close();self.assertTrue(admission.reservation_store.closed)

    def test_model_failure_and_disconnected_stream_release_claims(self):
        import struct
        entered,release,cleaned=threading.Event(),threading.Event(),threading.Event();self.events.append(release)
        class Runtime(FakeRuntime):
            def generate(self,*args,**kwargs):
                if kwargs.get('raw_prompt')=='fail':raise RuntimeError('injected model failure')
                if kwargs.get('raw_prompt')!='disconnect':yield from super().generate(*args,**kwargs);return
                try:
                    yield dict(text='hello')
                    entered.set()
                    if not release.wait(5):raise TimeoutError('test release missing')
                    yield dict(text='x'*65536)
                    yield from super().generate(*args,**kwargs)
                finally:cleaned.set()
        a,b=self.serve(Runtime()),self.serve();admission=a.RequestHandlerClass.client_admission
        self.assertEqual(self.call(a,prompt='fail')[0],500)
        wait_for(lambda:not admission.active);self.assertFalse(admission.reservation_store.claims)
        connection=socket.create_connection(('127.0.0.1',a.server_port),timeout=3)
        body=json.dumps(dict(model='a',prompt='disconnect',max_tokens=6,stream=True)).encode()
        wire=(f'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {self.client.key}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode()+body
        try:
            connection.sendall(wire);self.assertIn(b'200',connection.recv(1024));self.assertTrue(entered.wait(2))
            self.assertEqual(self.call(b)[0],429)
            connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        finally:connection.close();release.set()
        self.assertTrue(cleaned.wait(3));wait_for(lambda:not admission.active)
        self.assertFalse(admission.reservation_store.claims);self.assertEqual(self.call(b)[0],200)
