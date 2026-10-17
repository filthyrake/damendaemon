from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import tempfile
import threading
import unittest
from unittest.mock import patch
from damendaemon.http_auth import AuthPolicy,ClientKey
from damendaemon.host_tls import QUOTA_ALPN,client_context,server_context,certificate_pin
from damendaemon.remote_reservation_ledger import RemoteReservationLedger
from damendaemon.remote_quota_transport import RemoteQuotaClient,RemoteQuotaServer
from damendaemon.remote_quota_admission import RemoteQuotaReservations,RemoteQuotaRecoveryRequired
from damendaemon.shared_client_reservations import SharedReservationUnavailable,SharedReservationRejected
from damendaemon.server import BoundedHTTPServer,handler_for
from tests.test_server import FakeRuntime
from tests.test_distributed_drain import wait_for
from tests import test_host_tls as tls,test_shared_reservation_http as http_fixture


class RemoteQuotaAdmissionTests(unittest.TestCase):
    call=http_fixture.SharedReservationHttpTests.call
    blocker=http_fixture.SharedReservationHttpTests.blocker
    test_two_http_frontends_share_caps_and_client_isolation=http_fixture.SharedReservationHttpTests.test_two_http_servers_share_request_and_output_caps_with_client_isolation
    test_model_failure_and_disconnect_release_remote_claims=http_fixture.SharedReservationHttpTests.test_model_failure_and_disconnected_stream_release_claims

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)/'authority.sqlite'
        self.client=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=6)
        self.other=ClientKey('bob',secrets.token_urlsafe(32),1,max_reserved_output_tokens=3)
        self.admin=secrets.token_urlsafe(32);self.auth=AuthPolicy(admin_key=self.admin,clients=(self.client,self.other))
        self.ledger=RemoteReservationLedger(self.path,self.auth.clients,create=True)
        self.client_context=client_context(ca_file=tls.FIXTURES/'ca.pem',cert_file=tls.FIXTURES/'client.pem',key_file=tls.FIXTURES/'client.key',alpn=QUOTA_ALPN)
        self.server_context=server_context(ca_file=tls.FIXTURES/'ca.pem',cert_file=tls.FIXTURES/'server.pem',key_file=tls.FIXTURES/'server.key',alpn=QUOTA_ALPN)
        self.authorities=[];self.servers=[];self.events=[]
        self.start_authority()
        def cleanup():
            for event in self.events:event.set()
            for server,thread in reversed(self.servers):
                server.shutdown();thread.join(3)
                admission=server.RequestHandlerClass.client_admission;wait_for(lambda:not admission.active and not admission.reserved_output)
                admission.retry_releases();server.server_close()
            for server,thread in reversed(self.authorities):server.close();thread.join(3)
        self.addCleanup(cleanup)

    def start_authority(self,address=('127.0.0.1',0)):
        self.authority=RemoteQuotaServer(address,self.ledger,self.server_context,(certificate_pin(tls.FIXTURES/'client.pem'),))
        thread=threading.Thread(target=self.authority.serve_forever,daemon=True);thread.start();self.authorities.append((self.authority,thread))

    def quota_client(self,owner=None,key=None,allowed=None):
        owner,key=owner or secrets.token_hex(16),key or secrets.token_bytes(32)
        self.ledger.register(owner,key,allowed or ['alice','bob'])
        return RemoteQuotaClient(self.authority.address,self.client_context,certificate_pin(tls.FIXTURES/'server.pem'),
            'localhost',self.ledger.identity,owner,key,timeout=2)

    def store(self,client=None,policies=None):return RemoteQuotaReservations(client or self.quota_client(),policies or self.auth.clients)

    def serve(self,runtime=None):
        store=self.store()
        with patch.dict('os.environ',{},clear=True):handler=handler_for(runtime or FakeRuntime(),self.auth,reservation_store=store)
        server=BoundedHTTPServer(('127.0.0.1',0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.servers.append((server,thread));return server

    def test_policy_prior_claims_retired_owner_and_handle_isolation(self):
        client=self.quota_client();store=self.store(client)
        changed=ClientKey('alice',self.client.key,3,max_reserved_output_tokens=6)
        with self.assertRaises(RemoteQuotaRecoveryRequired):self.store(client,[changed])
        claim=store.reserve(self.client,outputs=6)
        with self.assertRaises(RuntimeError):store.close()
        with self.assertRaises(RemoteQuotaRecoveryRequired):self.store(client)
        other=self.store()
        with self.assertRaises(ValueError):other.release(claim)
        store.release(claim);store.release(claim);store.close();other.close()
        reopened=self.store(client);self.assertEqual(reopened.sequence,3);reopened.close()
        self.ledger.execute(client.owner,client._credential,3,dict(op='retire'))
        with self.assertRaises(RemoteQuotaRecoveryRequired):self.store(client)
        limited=self.quota_client(allowed=['alice'])
        with self.assertRaises(RemoteQuotaRecoveryRequired):self.store(limited)

    def test_lost_reserve_replies_are_resolved_before_releasing_ended_http_leases(self):
        for lost_reserve in (1,2):
            server=self.serve();admission=server.RequestHandlerClass.client_admission;store=admission.reservation_store
            reply=self.authority._reply;count=[0];snapshots=[];execute=self.ledger.execute
            def record(*args,**kwargs):
                snapshots.append((args[2],kwargs.get('operation_id')));return execute(*args,**kwargs)
            def drop(connection,response,deadline):
                if response.get('result',{}).get('code')=='reserved':
                    count[0]+=1
                    if count[0]==lost_reserve:raise OSError('lost accepted reply')
                return reply(connection,response,deadline)
            with patch.object(self.authority,'_reply',drop),patch.object(self.ledger,'execute',record),patch.object(FakeRuntime,'generate',side_effect=AssertionError('must not generate')):
                status,body=self.call(server)
            self.assertEqual(status,503);self.assertNotIn(self.client.key,body);self.assertTrue(admission.pending_releases)
            original=store.pending.request;sequence=store.pending.sequence
            self.assertEqual(self.call(server)[0],503);self.assertIs(store.pending.request,original)
            with patch.object(self.ledger,'execute',record):admission.retry_releases()
            seen=[nonce for seq,nonce in snapshots if seq==sequence]
            self.assertEqual(len(seen),2);self.assertEqual(seen[0],seen[1])
            self.assertFalse(store.claims);self.assertIsNone(store.pending);self.assertEqual(self.call(server)[0],200)

    def test_lost_output_release_reply_suppresses_done_and_cleanup_retries_exactly(self):
        server=self.serve();admission=server.RequestHandlerClass.client_admission;store=admission.reservation_store
        reply=self.authority._reply;dropped=[False]
        def drop(connection,response,deadline):
            if response.get('result',{}).get('code')=='released' and not dropped[0]:dropped[0]=True;raise OSError('lost release reply')
            return reply(connection,response,deadline)
        with patch.object(self.authority,'_reply',drop):status,body=self.call(server,stream=True,max_tokens=6)
        self.assertEqual(status,200);self.assertIn('reservation accounting unavailable',body);self.assertNotIn('[DONE]',body)
        wait_for(lambda:not admission.active);self.assertEqual(store.pending.kind,'release')
        pending=store.pending;admission.retry_releases();self.assertTrue(pending.claim._released)
        self.assertFalse(store.claims);self.assertEqual(self.call(server)[0],200)

    def test_authority_restart_preserves_active_http_lease_and_reconnects(self):
        runtime,entered,release=self.blocker();a,b=self.serve(runtime),self.serve()
        store=a.RequestHandlerClass.client_admission.reservation_store
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(self.call,a,prompt='hold',max_tokens=6)
            self.assertTrue(entered.wait(2));address=self.authority.address;self.authority.close()
            self.ledger=RemoteReservationLedger(self.path,self.auth.clients);self.start_authority(address)
            self.assertEqual(self.ledger.status(store.client.owner,store.client._credential)['claims'],2)
            self.assertEqual(self.call(b)[0],429)
            release.set();self.assertEqual(future.result(5)[0],200)
        wait_for(lambda:not a.RequestHandlerClass.client_admission.active)
        self.assertFalse(store.claims);self.assertEqual(self.call(b)[0],200)

    def test_conflicting_controller_never_releases_another_requests_claim(self):
        client=self.quota_client();a,b=self.store(client),self.store(client)
        claim=a.reserve(self.client,outputs=6)
        with self.assertRaises(SharedReservationUnavailable) as error:b.reserve(self.client,outputs=6)
        candidate=error.exception.claim;self.assertEqual(candidate.id,claim.id);self.assertIsNot(candidate,claim)
        with self.assertRaises(SharedReservationUnavailable):b.release(candidate)
        with self.assertRaises(SharedReservationUnavailable):b.close()
        self.assertEqual(self.ledger.status(client.owner,client._credential)['claims'],1)
        a.release(claim);a.close()
        with self.assertRaises(SharedReservationUnavailable):b.release(candidate)
        self.assertEqual(self.ledger.status(client.owner,client._credential)['claims'],0)
        # b remains explicitly unresolved; dropping a local controller never
        # changes durable remote capacity, and no socket/file owner is retained.

    def test_rejected_unknown_reserve_returns_no_capacity_claim_and_close_retries_release(self):
        a,b=self.store(),self.store();first=a.reserve(self.client,outputs=6)
        reply=self.authority._reply;drop=[True]
        def lost_rejection(connection,response,deadline):
            if response.get('result',{}).get('code')=='output_limit' and drop[0]:drop[0]=False;raise OSError('lost rejection')
            return reply(connection,response,deadline)
        with patch.object(self.authority,'_reply',lost_rejection):
            with self.assertRaises(SharedReservationUnavailable) as error:b.reserve(self.client,outputs=1)
        b.release(error.exception.claim);self.assertFalse(b.claims);self.assertIsNone(b.pending)
        with patch.object(self.authority,'_reply',side_effect=OSError('lost release')):
            with self.assertRaises(SharedReservationUnavailable):a.release(first)
        self.assertTrue(a.claims);a.close();self.assertTrue(a.closed);self.assertFalse(a.claims);b.close()

    def test_remote_capacity_release_does_not_refund_rolling_rates(self):
        from damendaemon.client_admission import ClientAdmission,ClientRateLimitError,ClientOutputRateLimitError
        client=ClientKey('alice',self.client.key,2,max_reserved_output_tokens=6,max_requests_per_minute=1,max_output_tokens_per_minute=3)
        store=self.store(policies=[client]);admission=ClientAdmission(AuthPolicy(clients=(client,)),reservation_store=store)
        self.addCleanup(admission.close);headers=['Bearer '+client.key]
        with self.assertRaises(RuntimeError):
            with admission.lease(headers),admission.output_lease(headers,3):raise RuntimeError('execution failed')
        self.assertFalse(store.claims)
        with self.assertRaises(ClientRateLimitError):
            with admission.lease(headers):pass
        with self.assertRaises(ClientOutputRateLimitError):
            with admission.output_lease(headers,1):pass
        self.assertFalse(store.claims)

    def test_lost_post_response_release_writes_no_second_http_response(self):
        import socket
        server=self.serve();admission=server.RequestHandlerClass.client_admission;store=admission.reservation_store
        reply=self.authority._reply
        def drop(connection,response,deadline):
            result=response.get('result',{})
            if result.get('code')=='released' and result.get('sequence')==4:raise OSError('lost final request release')
            return reply(connection,response,deadline)
        connection=socket.create_connection(('127.0.0.1',server.server_port),timeout=3)
        body=json.dumps(dict(model='a',prompt='hello',max_tokens=2)).encode()
        wire=(f'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {self.client.key}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()+body
        try:
            with patch.object(self.authority,'_reply',drop):
                connection.sendall(wire);chunks=[]
                while True:
                    data=connection.recv(65536)
                    if not data:break
                    chunks.append(data)
            response=b''.join(chunks);self.assertEqual(response.count(b'HTTP/1.1'),1);self.assertIn(b'200 OK',response)
        finally:connection.close()
        self.assertEqual(store.pending.kind,'release');self.assertEqual(store.pending.sequence,4)
        admission.retry_releases();self.assertFalse(store.claims)
