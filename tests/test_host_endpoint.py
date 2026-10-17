from concurrent.futures import ThreadPoolExecutor
import socket
import ssl
import threading
import time
import unittest
from unittest.mock import patch
import torch
from damendaemon.host_endpoint import HostPeerListener,connect_peer
from damendaemon.host_tls import HostTlsCancelled,certificate_pin
from damendaemon.host_tensor_channel import HostTensorChannel
from tests.test_host_tls import contexts,FIXTURES


class HostEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client_context,self.server_context=contexts()
        self.client_pin=certificate_pin(FIXTURES/'client.pem');self.server_pin=certificate_pin(FIXTURES/'server.pem')

    def listening(self):
        raw=socket.socket();raw.bind(('127.0.0.1',0));raw.listen(4);self.addCleanup(raw.close)
        return raw

    def owner(self):
        raw=self.listening();address=raw.getsockname()
        owner=HostPeerListener(raw,self.server_context,peer_pin=self.client_pin);self.addCleanup(owner.close)
        return owner,address

    def connect(self,address,**options):
        return connect_peer(*address,self.client_context,peer_pin=self.server_pin,server_hostname='localhost',**options)

    def test_authenticated_bootstrap_transfers_channels_and_owner_close_keeps_peer(self):
        owner,address=self.owner()
        with ThreadPoolExecutor(2) as pool:
            future=pool.submit(owner.accept,timeout=1)
            with patch('socket.getaddrinfo',side_effect=AssertionError('DNS')):
                client=self.connect(address,timeout=1)
            server=future.result(timeout=2)
            for peer in (client,server):self.addCleanup(peer.close);self.assertIsInstance(peer,ssl.SSLSocket);self.assertIsNone(peer.gettimeout())
            owner.close();channels=[HostTensorChannel(peer,b'x'*32,rank,timeout=1) for rank,peer in enumerate((client,server))]
            for channel in channels:self.addCleanup(channel.close)
            list(pool.map(lambda c:c.handshake(),channels))
            expected=torch.arange(12,dtype=torch.float32).reshape(3,4)
            channels[0].send(expected);torch.testing.assert_close(channels[1].receive((3,4),torch.float32),expected)

    def test_accept_timeout_and_listener_close_interrupt_without_peer(self):
        owner,_=self.owner();start=time.monotonic()
        with self.assertRaises(TimeoutError):owner.accept(timeout=.03)
        self.assertLess(time.monotonic()-start,.5);self.assertFalse(owner.closed.is_set())
        with ThreadPoolExecutor(1) as pool:
            entered=threading.Event()
            def cancel():entered.set();return False
            future=pool.submit(owner.accept,timeout=5,cancelled=cancel)
            self.assertTrue(entered.wait(1));start=time.monotonic();owner.close()
            with self.assertRaises(HostTlsCancelled):future.result(timeout=1)
            self.assertLess(time.monotonic()-start,.5)
        with self.assertRaises(HostTlsCancelled):owner.accept(timeout=1)

    def test_close_stalled_tls_and_reject_concurrent_accept(self):
        owner,address=self.owner();entered=threading.Event();original=self.server_context.wrap_socket;wrapped=[]
        def wrapping(*args,**kwargs):
            value=original(*args,**kwargs);wrapped.append(value);entered.set();return value
        with ThreadPoolExecutor(1) as pool,patch.object(self.server_context,'wrap_socket',wrapping):
            future=pool.submit(owner.accept,timeout=5)
            peer=socket.create_connection(address);self.addCleanup(peer.close)
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(RuntimeError,'pending'):owner.accept(timeout=1)
            start=time.monotonic();owner.close()
            with self.assertRaises(HostTlsCancelled):future.result(timeout=1)
            self.assertLess(time.monotonic()-start,.5)
        self.assertTrue(wrapped);self.assertTrue(all(s.fileno()==-1 for s in wrapped))

    def test_connector_cancel_stalled_tls_closes_owned_socket(self):
        raw=self.listening();address=raw.getsockname();cancelled=threading.Event();entered=threading.Event();wrapped=[]
        original=self.client_context.wrap_socket
        def wrapping(*args,**kwargs):
            value=original(*args,**kwargs);wrapped.append(value);entered.set();return value
        with ThreadPoolExecutor(1) as pool,patch.object(self.client_context,'wrap_socket',wrapping):
            future=pool.submit(self.connect,address,timeout=5,cancelled=cancelled.is_set)
            peer,_=raw.accept();self.addCleanup(peer.close);self.assertTrue(entered.wait(1));start=time.monotonic();cancelled.set()
            with self.assertRaises(HostTlsCancelled):future.result(timeout=1)
            self.assertLess(time.monotonic()-start,.5)
        self.assertTrue(all(s.fileno()==-1 for s in wrapped))

    def test_wrong_pin_rejects_and_listener_remains_reusable(self):
        owner,address=self.owner();owner.peer_pin=b'x'*32
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(owner.accept,timeout=1)
            client=self.connect(address,timeout=1);self.addCleanup(client.close)
            with self.assertRaisesRegex(ValueError,'pin'):future.result(timeout=2)
            owner.peer_pin=self.client_pin
            future=pool.submit(owner.accept,timeout=1)
            client=self.connect(address,timeout=1);self.addCleanup(client.close)
            server=future.result(timeout=2);self.addCleanup(server.close)
        self.assertFalse(owner.closed.is_set())

    def test_invalid_inputs_remain_caller_owned_and_no_dns(self):
        raw=self.listening()
        with self.assertRaises(ValueError):HostPeerListener(raw,self.server_context,peer_pin=b'bad')
        self.assertGreaterEqual(raw.fileno(),0)
        invalid=socket.socket();self.addCleanup(invalid.close)
        with self.assertRaises(ValueError):HostPeerListener(invalid,self.server_context,peer_pin=self.client_pin)
        self.assertGreaterEqual(invalid.fileno(),0)
        with patch('socket.getaddrinfo',side_effect=AssertionError('DNS')),patch('damendaemon.host_endpoint.socket.socket',side_effect=AssertionError('allocated')):
            for address,port in (('localhost',42),('127.0.0.1',True),('127.0.0.1',0),('::1%lo0',42),('',42)):
                with self.assertRaises(ValueError):self.connect((address,port),timeout=1)
            with self.assertRaises(HostTlsCancelled):self.connect(('127.0.0.1',42),timeout=1,cancelled=lambda:True)
        owner,_=self.owner()
        for options in (dict(timeout=True),dict(timeout=float('nan')),dict(cancelled=3)):
            with self.assertRaises(ValueError):owner.accept(**options)

    def test_bound_socket_is_started_and_connected_socket_rejected(self):
        raw=socket.socket();self.addCleanup(raw.close);raw.bind(('127.0.0.1',0));address=raw.getsockname()
        owner=HostPeerListener(raw,self.server_context,peer_pin=self.client_pin,backlog=2);self.addCleanup(owner.close)
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(owner.accept,timeout=1);client=self.connect(address,timeout=1);server=future.result(timeout=2)
            self.addCleanup(client.close);self.addCleanup(server.close)
        plain=self.listening();client=socket.create_connection(plain.getsockname());server,_=plain.accept()
        self.addCleanup(client.close);self.addCleanup(server.close)
        with self.assertRaises(ValueError):HostPeerListener(server,self.server_context,peer_pin=self.client_pin)
        self.assertGreaterEqual(server.fileno(),0)
        for backlog in (0,17,True):
            with self.assertRaises(ValueError):HostPeerListener(plain,self.server_context,peer_pin=self.client_pin,backlog=backlog)
            self.assertGreaterEqual(plain.fileno(),0)

    def test_refused_connection_and_combined_accept_handshake_deadline(self):
        raw=self.listening();address=raw.getsockname();raw.close()
        with self.assertRaises(OSError):self.connect(address,timeout=.2)
        owner,address=self.owner();entered=threading.Event();original=self.server_context.wrap_socket;wrapped=[]
        def wrapping(*args,**kwargs):
            value=original(*args,**kwargs);wrapped.append(value);entered.set();return value
        with ThreadPoolExecutor(1) as pool,patch.object(self.server_context,'wrap_socket',wrapping):
            start=time.monotonic();future=pool.submit(owner.accept,timeout=.15)
            # Deliberately spend part of the one deadline waiting for TCP.
            threading.Event().wait(.04)
            peer=socket.create_connection(address);self.addCleanup(peer.close);self.assertTrue(entered.wait(1))
            with self.assertRaises(TimeoutError):future.result(timeout=1)
            elapsed=time.monotonic()-start;self.assertGreater(elapsed,.1);self.assertLess(elapsed,.4)
        self.assertTrue(all(s.fileno()==-1 for s in wrapped));self.assertFalse(owner.closed.is_set())
