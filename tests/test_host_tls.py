from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import ssl
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from damendaemon.host_tls import ALPN,client_context,server_context,certificate_pin,wrap_peer
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.stage_protocol import StageState,Command,PREFILL

from tests.tls_credentials import credentials

FIXTURES=credentials()


def contexts(client='client'):
    return [client_context(ca_file=FIXTURES/'ca.pem',cert_file=FIXTURES/(client+'.pem'),key_file=FIXTURES/(client+'.key')),
            server_context(ca_file=FIXTURES/'ca.pem',cert_file=FIXTURES/'server.pem',key_file=FIXTURES/'server.key')]


class HostTlsTests(unittest.TestCase):
    def raw_pair(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen(1)
            left=socket.create_connection(listener.getsockname(),timeout=1);right,_=listener.accept()
        for item in (left,right):self.addCleanup(item.close)
        return left,right

    def pair(self,*,configured=None,hostname='localhost',pins=None):
        configured=contexts() if configured is None else configured
        pins=[certificate_pin(FIXTURES/'server.pem'),certificate_pin(FIXTURES/'client.pem')] if pins is None else pins
        raw=self.raw_pair();result=[]
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(wrap_peer,sock,context,server_side=bool(rank),peer_pin=pin,
                                 server_hostname=hostname if rank==0 else None,timeout=.5)
                     for rank,(sock,context,pin) in enumerate(zip(raw,configured,pins))]
            for future in futures:
                try:
                    value=future.result(timeout=2);self.addCleanup(value.close)
                except Exception as error:value=error
                result.append(value)
        for sock in raw:self.assertEqual(sock.fileno(),-1)
        return result

    def test_mutual_tls_carries_tensor_and_stage_frames(self):
        sockets=self.pair()
        for sock in sockets:
            self.assertIsInstance(sock,ssl.SSLSocket);self.assertEqual(sock.version(),'TLSv1.3');self.assertEqual(sock.selected_alpn_protocol(),ALPN)
        transports=[HostTensorChannel(sock,b'x'*32,rank,max_frame_bytes=4096,timeout=1) for rank,sock in enumerate(sockets)]
        channels=[HostStageChannel(t,StageState(7,8),vocab_size=100) for t in transports]
        for channel in channels:self.addCleanup(channel.close)
        def together(functions):
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(f) for f in functions]
                return [f.result(timeout=3) for f in futures]
        together([c.handshake for c in channels])
        command=Command(7,1,PREFILL,3,((0,1,0),))
        self.assertEqual(together([lambda:channels[0].begin(command),channels[1].begin]),[command,command])
        activation=torch.arange(12,dtype=torch.float32).reshape(3,4)
        transports[0].send(activation);torch.testing.assert_close(transports[1].receive((3,4),torch.float32),activation)
        results=together([lambda:channels[0].complete_tokens(command),lambda:channels[1].complete_tokens(command,torch.tensor([9]))])
        for result in results:self.assertEqual(result.tolist(),[9])
        self.assertEqual(channels[0].state.lengths,channels[1].state.lengths)

    def test_wrong_hostname_pin_or_alpn_rejected_without_fallback(self):
        result=self.pair(hostname='wrong.invalid');self.assertIsInstance(result[0],ssl.SSLCertVerificationError)
        result=self.pair(pins=[b'0'*32,certificate_pin(FIXTURES/'client.pem')])
        self.assertIsInstance(result[0],ValueError);self.assertIn('pin',str(result[0]))
        configured=contexts();configured[0].set_alpn_protocols(['other-protocol'])
        result=self.pair(configured=configured)
        for value in result:self.assertIsInstance(value,ValueError);self.assertIn('protocol',str(value))

    def test_server_requires_trusted_client_certificate(self):
        result=self.pair(configured=contexts('untrusted'))
        self.assertIsInstance(result[1],ssl.SSLCertVerificationError)
        configured=contexts();empty=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        empty.minimum_version=ssl.TLSVersion.TLSv1_3;empty.load_verify_locations(cafile=FIXTURES/'ca.pem');empty.set_alpn_protocols([ALPN])
        configured[0]=empty;result=self.pair(configured=configured)
        self.assertIsInstance(result[1],ssl.SSLError)
        # TLS 1.3 may finish the client's handshake before it receives the
        # server's missing-certificate alert. No framed session can then start.
        if isinstance(result[0],ssl.SSLSocket):
            channel=HostTensorChannel(result[0],b'x'*32,0,timeout=.2);self.addCleanup(channel.close)
            with self.assertRaises((OSError,EOFError)):channel.handshake()
            self.assertTrue(channel.closed)

    def test_stalled_handshake_deadline_closes_owned_descriptor(self):
        left,right=self.raw_pair();context=contexts()[0];wrapped=[];original=context.wrap_socket
        def capture(*args,**kwargs):
            value=original(*args,**kwargs);wrapped.append(value);return value
        started=time.monotonic()
        with patch.object(context,'wrap_socket',side_effect=capture):
            with self.assertRaises(TimeoutError):wrap_peer(left,context,server_side=False,peer_pin=b'a'*32,server_hostname='localhost',timeout=.03)
        self.assertLess(time.monotonic()-started,1)
        self.assertEqual(left.fileno(),-1);self.assertEqual(wrapped[0].fileno(),-1)

    def test_unsafe_local_options_leave_socket_with_caller(self):
        left,right=self.raw_pair();context=contexts()[0]
        for change in ({'peer_pin':b'bad'},{'timeout':True},{'server_hostname':None},{'timeout':float('nan')}):
            options=dict(server_side=False,peer_pin=b'a'*32,server_hostname='localhost')|change
            with self.assertRaises(ValueError):wrap_peer(left,context,**options)
            self.assertGreaterEqual(left.fileno(),0)
        context.check_hostname=False;context.verify_mode=ssl.CERT_NONE
        with self.assertRaises(ValueError):wrap_peer(left,context,server_side=False,peer_pin=b'a'*32,server_hostname='localhost')
        self.assertGreaterEqual(left.fileno(),0)
        context=contexts()[0];context.minimum_version=ssl.TLSVersion.TLSv1_2
        with self.assertRaises(ValueError):wrap_peer(left,context,server_side=False,peer_pin=b'a'*32,server_hostname='localhost')

    def test_pin_input_is_bounded_and_public_certificate_only(self):
        self.assertEqual(len(certificate_pin(FIXTURES/'server.pem')),32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'oversized.pem';path.write_bytes(b'x'*65537)
            with self.assertRaisesRegex(ValueError,'size'):certificate_pin(path)

    def test_channel_close_interrupts_tls_receive(self):
        import threading
        sockets=self.pair();channels=[HostTensorChannel(s,b'x'*32,i,timeout=10) for i,s in enumerate(sockets)]
        for channel in channels:self.addCleanup(channel.close)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda c:c.handshake(),channels))
            entered=threading.Event();receive=channels[1].connection.recv_into
            def waiting(*args,**kwargs):
                entered.set();return receive(*args,**kwargs)
            with patch.object(channels[1].connection,'recv_into',side_effect=waiting):
                future=pool.submit(channels[1].receive,(1,),torch.float32)
                self.assertTrue(entered.wait(1));started=time.monotonic();channels[1].close()
                with self.assertRaises((OSError,EOFError,RuntimeError)):future.result(timeout=1)
                self.assertLess(time.monotonic()-started,1)
