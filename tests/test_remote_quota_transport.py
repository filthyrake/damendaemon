from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch
from dllm.host_tls import QUOTA_ALPN,client_context,server_context,certificate_pin,wrap_peer
from dllm.remote_quota_transport import RemoteQuotaServer,RemoteQuotaClient,RemoteQuotaUnavailable,RemoteQuotaRejected,QuotaRequest,MAX_REQUEST,_json,_send,_receive
from tests import test_host_tls as tls,test_remote_reservation_ledger as ledger_fixture
from tests.test_distributed_drain import wait_for


class RemoteQuotaTransportTests(unittest.TestCase):
    def setUp(self):
        fixture=ledger_fixture.RemoteReservationLedgerTests();fixture.setUp();self.addCleanup(fixture.doCleanups);self.fixture=fixture
        self.owner,self.key=fixture.owner,fixture.key;self.ledger=fixture.ledger
        self.client_context,self.server_context=self.contexts()
        self.server_pin=certificate_pin(tls.FIXTURES/'server.pem');self.client_pin=certificate_pin(tls.FIXTURES/'client.pem')
        self.servers=[]
        def cleanup():
            for server,thread in reversed(self.servers):server.close();thread.join(3)
        self.addCleanup(cleanup)

    def contexts(self):
        return [client_context(ca_file=tls.FIXTURES/'ca.pem',cert_file=tls.FIXTURES/'client.pem',key_file=tls.FIXTURES/'client.key',alpn=QUOTA_ALPN),
                server_context(ca_file=tls.FIXTURES/'ca.pem',cert_file=tls.FIXTURES/'server.pem',key_file=tls.FIXTURES/'server.key',alpn=QUOTA_ALPN)]

    def serve(self,ledger=None,**options):
        server=RemoteQuotaServer(('127.0.0.1',0),ledger or self.ledger,self.server_context,(self.client_pin,),**options)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.servers.append((server,thread))
        return server

    def client(self,server,**options):
        return RemoteQuotaClient(server.address,self.client_context,options.pop('peer_pin',self.server_pin),'localhost',
            options.pop('ledger',server.ledger.identity),options.pop('owner',self.owner),options.pop('credential',self.key),**options)

    def connected(self,server):
        raw=socket.create_connection(server.address,timeout=2)
        connection=wrap_peer(raw,self.client_context,server_side=False,peer_pin=self.server_pin,server_hostname='localhost',alpn=QUOTA_ALPN,timeout=1)
        self.addCleanup(connection.close);return connection

    def test_real_tls_operations_status_pages_and_exact_retry(self):
        server=self.serve();client=self.client(server)
        command=ledger_fixture.reserve(8);request=client.request('execute',sequence=1,command=command);command['outputs']=1
        result=client.call(request);self.assertTrue(result['ok']);self.assertEqual(client.call(request),result)
        other_request=client.request('execute',sequence=1,command=ledger_fixture.reserve(8))
        with self.assertRaises(RemoteQuotaRejected) as error:client.call(other_request)
        self.assertEqual(error.exception.code,'sequence_conflict');self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)
        self.assertNotIn(self.key.hex(),repr(request));self.assertNotIn(self.key.hex(),repr(client))
        status=client.call(client.request('status'));self.assertEqual((status['claims'],status['sequence']),(1,1))
        page=client.call(client.request('list',after=None,limit=1));self.assertEqual(page['claims'][0]['outputs'],8)
        resized=client.call(client.request('execute',sequence=2,command=dict(op='resize',claim=result['claim'],outputs=10)));self.assertTrue(resized['ok'])
        client.call(client.request('execute',sequence=3,command=dict(op='release',claim=result['claim'])))
        self.assertEqual(client.call(client.request('status'))['claims'],0)

    def test_response_loss_after_commit_and_authority_restart_keep_charge(self):
        server=self.serve();client=self.client(server);request=client.request('execute',sequence=1,command=ledger_fixture.reserve(10))
        with patch.object(server,'_reply',side_effect=OSError('lost reply')):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(request)
        result=client.call(request);self.assertEqual(client.call(client.request('status'))['claims'],1)
        server.close();restarted=self.serve(self.fixture.reopen());client=self.client(restarted)
        self.assertEqual(client.call(request),result)
        owner,key=secrets.token_hex(16),secrets.token_bytes(32);restarted.ledger.register(owner,key,['alice'])
        other=self.client(restarted,owner=owner,credential=key)
        self.assertEqual(other.call(other.request('execute',sequence=1,command=ledger_fixture.reserve(1)))['code'],'output_limit')
        client.call(client.request('execute',sequence=2,command=dict(op='release',claim=result['claim'])))
        self.assertTrue(other.call(other.request('execute',sequence=2,command=ledger_fixture.reserve(10)))['ok'])

    def test_late_receipt_validation_is_unknown_and_exact_retry_keeps_one_charge(self):
        server=self.serve();client=self.client(server,timeout=1)
        request=client.request('execute',sequence=1,command=ledger_fixture.reserve(8))
        monotonic=time.monotonic;offset=[0.0];validate=client._result
        def delayed(*args):
            validate(*args)
            offset[0]=2.0
        with patch('dllm.remote_quota_transport.time.monotonic',side_effect=lambda:monotonic()+offset[0]), \
                patch.object(client,'_result',side_effect=delayed):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(request)
        self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)
        result=client.call(request)
        self.assertTrue(result['ok'])
        self.assertEqual(result['sequence'],1)
        self.assertEqual(client.call(request),result)
        self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)

    def test_wrong_pin_protocol_credentials_and_ledger_never_execute(self):
        server=self.serve()
        with patch.object(self.ledger,'execute',side_effect=AssertionError('must not execute')) as execute:
            client=self.client(server,peer_pin=secrets.token_bytes(32))
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
            client=self.client(server,credential=secrets.token_bytes(32))
            with self.assertRaises(RemoteQuotaRejected) as error:client.call(client.request('status'))
            self.assertEqual(error.exception.code,'owner_denied')
            client=self.client(server,ledger=secrets.token_hex(16))
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('execute',sequence=1,command=ledger_fixture.reserve()))
            context=tls.contexts()[0]
            client=RemoteQuotaClient(server.address,context,self.server_pin,'localhost',self.ledger.identity,self.owner,self.key,timeout=.5)
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
            execute.assert_not_called()
        wrong=RemoteQuotaServer(('127.0.0.1',0),self.fixture.reopen(),self.server_context,(secrets.token_bytes(32),))
        thread=threading.Thread(target=wrong.serve_forever,daemon=True);thread.start();self.servers.append((wrong,thread))
        client=self.client(wrong)
        with patch.object(wrong,'_dispatch',side_effect=AssertionError('untrusted peer')) as dispatch:
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
            dispatch.assert_not_called()
        self.assertEqual(self.ledger.status(self.owner,self.key)['sequence'],0)

    def test_plaintext_oversized_duplicate_and_unknown_methods_fail_closed(self):
        server=self.serve(timeout=.3);client=self.client(server)
        raw=socket.create_connection(server.address,timeout=1);raw.sendall(b'GET / HTTP/1.0\r\n\r\n');raw.close()
        connection=self.connected(server);connection.sendall(struct.pack('!I',MAX_REQUEST+1))
        try:self.assertEqual(_receive(connection,65536,time.monotonic()+1)['error'],'invalid_request')
        finally:connection.close()
        for mutation in ('duplicate','enroll','unknown_field'):
            data=json.loads(client.request('status').payload)
            if mutation=='enroll':data['method']='register'
            if mutation=='unknown_field':data['unexpected']=True
            payload=json.dumps(data).encode()
            if mutation=='duplicate':payload=payload[:-1]+b',"version":1}'
            connection=self.connected(server)
            try:
                _send(connection,payload,time.monotonic()+1);response=_receive(connection,65536,time.monotonic()+1)
                self.assertEqual(response['error'],'invalid_request')
            finally:connection.close()
        self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],0)

    def test_saturation_handshake_and_partial_frame_deadlines_release_workers(self):
        server=self.serve(max_connections=1,timeout=.2)
        stalled=socket.create_connection(server.address,timeout=1);self.addCleanup(stalled.close)
        wait_for(lambda:len(server.workers)==1)
        client=self.client(server,timeout=.3)
        with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
        wait_for(lambda:not server.workers)
        connection=self.connected(server);connection.sendall(struct.pack('!I',100)+b'{')
        wait_for(lambda:len(server.workers)==1);wait_for(lambda:not server.workers);connection.close()
        self.assertEqual(client.call(client.request('status'))['claims'],0)

    def test_shutdown_interrupts_socket_and_retains_blocked_cleanup_for_retry(self):
        server=self.serve();client=self.client(server);entered,release=threading.Event(),threading.Event()
        self.addCleanup(release.set);dispatch=server._dispatch
        def blocked(body):
            result=dispatch(body);entered.set();release.wait(5);return result
        with ThreadPoolExecutor(1) as pool:
            with patch.object(server,'_dispatch',blocked):
                future=pool.submit(client.call,client.request('execute',sequence=1,command=ledger_fixture.reserve(10)))
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(RuntimeError,'cleanup'):server.close()
                self.assertFalse(server.closed);self.assertFalse(self.ledger.closed);self.assertTrue(server.workers)
                release.set()
                with self.assertRaises(RemoteQuotaUnavailable):future.result(3)
                wait_for(lambda:not server.workers);server.close();self.assertTrue(server.closed)
        restarted=self.fixture.reopen();self.assertEqual(restarted.status(self.owner,self.key)['claims'],1)

    def test_client_deadline_and_bad_response_retain_unknown_outcome(self):
        server=self.serve();client=self.client(server,timeout=.1);reply=server._reply
        def delayed(*args):time.sleep(.2);return reply(*args)
        request=client.request('execute',sequence=1,command=ledger_fixture.reserve(10))
        with patch.object(server,'_reply',delayed):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(request)
        client=self.client(server);self.assertTrue(client.call(request)['ok'])
        def wrong(connection,response,deadline):_send(connection,_json(dict(version=1,ledger='wrong',result={}),65536),deadline)
        with patch.object(server,'_reply',wrong):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
        self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)

    def test_invalid_local_options_leave_ledger_owned_by_caller(self):
        for options in (dict(max_connections=True),dict(max_connections=129),dict(timeout=0),dict(timeout=float('nan'))):
            with self.assertRaises(ValueError):RemoteQuotaServer(('127.0.0.1',0),self.ledger,self.server_context,(self.client_pin,),**options)
        for pins in ((),(self.client_pin,self.client_pin),(b'bad',)):
            with self.assertRaises(ValueError):RemoteQuotaServer(('127.0.0.1',0),self.ledger,self.server_context,pins)
        self.assertFalse(self.ledger.closed)
        with self.assertRaises(ValueError):RemoteQuotaClient(('localhost',1234),self.client_context,self.server_pin,'localhost',self.ledger.identity,self.owner,self.key)
        server=self.serve();client=self.client(server)
        with self.assertRaises(ValueError):client.request('register')
        with self.assertRaises(ValueError):client.call(QuotaRequest(b'{}'))

    def test_malformed_success_receipts_and_pages_are_unknown_outcomes(self):
        server=self.serve();client=self.client(server)
        request=client.request('execute',sequence=1,command=ledger_fixture.reserve(1))
        def reply(result):
            def send(connection,response,deadline):
                _send(connection,_json(dict(version=1,ledger=self.ledger.identity,operation_id=response.get('operation_id'),result=result),65536),deadline)
            return send
        for result in ({},dict(sequence=1,ok=True,code='reserved',claim=secrets.token_hex(16)+':1'),
                       dict(sequence=1,ok=True,code='released',claim=self.owner+':1'),
                       dict(sequence=1,ok=False,code='reserved',claim=self.owner+':1')):
            with patch.object(server,'_reply',reply(result)):
                with self.assertRaises(RemoteQuotaUnavailable):client.call(request)
        self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)
        with patch.object(server,'_reply',reply(dict(claims=[],after=self.owner+':1'))):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('list',after=None,limit=1))
        with patch.object(server,'_reply',reply(dict(ledger=self.ledger.identity,sequence=1,retired=False,claims=-1,receipt=None))):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
        def swapped(connection,response,deadline):
            response['operation_id']=secrets.token_hex(16)
            _send(connection,_json(dict(version=1,ledger=self.ledger.identity,**response),65536),deadline)
        with patch.object(server,'_reply',swapped):
            with self.assertRaises(RemoteQuotaUnavailable):client.call(request)
        self.assertTrue(client.call(request)['ok'])

    def test_multiple_pins_accept_only_configured_certificates(self):
        server=RemoteQuotaServer(('127.0.0.1',0),self.ledger,self.server_context,(secrets.token_bytes(32),self.client_pin))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.servers.append((server,thread))
        self.assertEqual(self.client(server).call(self.client(server).request('status'))['claims'],0)
        client_context=self.client_context
        self.client_context=tls.contexts('untrusted')[0];self.client_context.set_alpn_protocols([QUOTA_ALPN])
        try:
            client=self.client(server)
            with self.assertRaises(RemoteQuotaUnavailable):client.call(client.request('status'))
        finally:self.client_context=client_context
