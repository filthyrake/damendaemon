from concurrent.futures import ThreadPoolExecutor, wait
import socket
import time
import unittest
from unittest.mock import patch

import torch

from dllm.host_tensor_channel import HostTensorChannel,FRAME,MAGIC,HELLO,VERSION,DTYPES


class HostTensorChannelTests(unittest.TestCase):
    def pair(self,*,caps=(4096,2048),identities=(b'a'*32,b'a'*32),timeout=1):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        try:
            left=socket.create_connection(listener.getsockname(),timeout=1)
            right,_=listener.accept()
        finally:listener.close()
        channels=[HostTensorChannel(sock,identity,rank,max_frame_bytes=cap,timeout=timeout)
                  for rank,(sock,identity,cap) in enumerate(zip((left,right),identities,caps))]
        for channel in channels:self.addCleanup(channel.close)
        return channels

    def ready(self,**kwargs):
        channels=self.pair(**kwargs)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda channel:channel.handshake(),channels))
        return channels

    def test_exact_dtype_shape_sequence_and_owned_payloads(self):
        left,right=self.ready()
        self.assertEqual(left.max_frame_bytes,2048);self.assertEqual(right.max_frame_bytes,2048)
        received=[]
        for dtype in DTYPES.values():
            source=torch.arange(24).reshape(2,3,4).to(dtype)
            with ThreadPoolExecutor(1) as pool:
                sending=pool.submit(left.send,source)
                result=right.receive(source.shape,dtype)
                sending.result()
            torch.testing.assert_close(result,source,rtol=0,atol=0)
            received.append(result)
            source.zero_()
            self.assertEqual(int(result.sum()),276)
        received[-1].zero_()
        self.assertEqual(int(received[0].sum()),276)
        right.send(torch.tensor([9],dtype=torch.int64))
        self.assertEqual(left.receive((1,),torch.int64).tolist(),[9])
        self.assertEqual((left.sent,right.received,left.received,right.sent),(6,6,1,1))

    def test_identity_mismatch_closes_both_owned_sockets(self):
        channels=self.pair(identities=(b'a'*32,b'b'*32))
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(channel.handshake) for channel in channels]
            for future in futures:
                with self.assertRaises(ValueError):future.result()
        for channel in channels:
            self.assertTrue(channel.failed);self.assertEqual(channel.connection.fileno(),-1)

    def test_metadata_rejected_before_payload_read(self):
        canonical=[MAGIC,1,1,1,0,4,0,0,0,16]
        for index,value in ((0,b'BADFRAME'),(1,2),(2,2),(3,2),(4,1),(5,8),(6,1),(9,2**63)):
            left,right=self.ready()
            header=canonical.copy();header[index]=value
            left.connection.sendall(FRAME.pack(*header))
            with patch.object(right,'_read',wraps=right._read) as read:
                with self.assertRaises(ValueError):right.receive((4,),torch.float32)
                self.assertEqual([call.args[0] for call in read.call_args_list],[FRAME.size])
            self.assertTrue(right.failed);self.assertEqual(right.received,0)
            left.close()

    def test_truncated_payload_and_timeout_poison_connection(self):
        left,right=self.ready()
        left.connection.sendall(FRAME.pack(MAGIC,1,1,1,0,4,0,0,0,16)+b'123')
        left.close()
        with self.assertRaises(EOFError):right.receive((4,),torch.float32)
        self.assertTrue(right.failed)
        left,right=self.ready(timeout=.05)
        started=time.monotonic()
        with self.assertRaises(TimeoutError):right.receive((1,),torch.int64)
        self.assertLess(time.monotonic()-started,1)
        self.assertTrue(right.closed)

    def test_local_invalid_layout_does_not_write_or_poison(self):
        left,right=self.ready(caps=(16,16))
        for tensor in (torch.zeros(5),torch.zeros(2,3).T,torch.zeros(1,dtype=torch.bool),torch.zeros(0)):
            with self.assertRaises(ValueError):left.send(tensor)
        self.assertFalse(left.failed);self.assertEqual(left.sent,0)
        left.send(torch.tensor([1,2,3,4],dtype=torch.float32))
        self.assertEqual(right.receive((4,),torch.float32).tolist(),[1,2,3,4])
        with self.assertRaises(RuntimeError):left.handshake()

    def test_close_interrupts_blocked_receive_and_prevents_commit(self):
        import threading
        left,right=self.ready(timeout=10)
        entered=threading.Event()
        original=right._read
        def blocked(*args):
            entered.set()
            return original(*args)
        with patch.object(right,'_read',side_effect=blocked),ThreadPoolExecutor(1) as pool:
            receiving=pool.submit(right.receive,(1,),torch.int64)
            self.assertTrue(entered.wait(1))
            right.close()
            done,_=wait([receiving],timeout=.5)
            self.assertIn(receiving,done,'close must interrupt receive before its operation deadline')
            with self.assertRaises((EOFError,OSError,RuntimeError)):receiving.result()
        self.assertTrue(right.closed);self.assertEqual(right.received,0)
        self.assertEqual(right.connection.fileno(),-1)

    def test_one_deadline_spans_fragmented_reads(self):
        left,right=self.ready(timeout=.05)
        left.connection.sendall(FRAME.pack(MAGIC,1,1,1,0,1,0,0,0,4)+b'1234')
        connection=right.connection
        class Fragmented:
            def __getattr__(self,name):return getattr(connection,name)
            def recv_into(self,view):return connection.recv_into(view[:1])
        right.connection=Fragmented()
        # Header fragments arrive, but they cannot reset the overall deadline.
        with patch('dllm.host_tensor_channel.time.monotonic',side_effect=[0,.01,.02,.06]):
            with self.assertRaises(TimeoutError):right.receive((1,),torch.float32)
        self.assertTrue(right.failed);self.assertEqual(right.received,0)

    def test_late_final_operations_never_commit_success(self):
        import sys
        for operation in ('handshake','send','receive'):
            with self.subTest(operation=operation):
                left,right = self.pair() if operation=='handshake' else self.ready()
                clock=[0.0]
                target=left if operation!='receive' else right
                if operation=='handshake':
                    right.connection.sendall(HELLO.pack(MAGIC,1,0 if sys.byteorder=='little' else 1,VERSION,4096,b'a'*32))
                    original=target._read
                    def delayed(*args):
                        result=original(*args);clock[0]=2.0;return result
                    instrument=patch.object(target,'_read',side_effect=delayed)
                    action=target.handshake
                elif operation=='send':
                    original=target._send
                    def delayed(data,deadline):
                        original(data,deadline)
                        if len(data)!=FRAME.size:clock[0]=2.0
                    instrument=patch.object(target,'_send',side_effect=delayed)
                    action=lambda:target.send(torch.ones(1))
                else:
                    left.send(torch.ones(1))
                    original=torch.frombuffer
                    def delayed(*args,**kwargs):
                        result=original(*args,**kwargs);clock[0]=2.0;return result
                    instrument=patch('dllm.host_tensor_channel.torch.frombuffer',side_effect=delayed)
                    action=lambda:target.receive((1,),torch.float32)
                with patch('dllm.host_tensor_channel.time.monotonic',side_effect=lambda:clock[0]),instrument:
                    with self.assertRaises(TimeoutError):action()
                self.assertTrue(target.failed)
                self.assertTrue(target.closed)
                self.assertFalse(target.ready)
                self.assertEqual((target.sent,target.received),(0,0))
                self.assertEqual(target.connection.fileno(),-1)

    def test_send_interruption_closes_owned_socket(self):
        left,right=self.ready()
        with patch.object(left,'_send',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):left.send(torch.ones(1))
        self.assertTrue(left.failed);self.assertEqual(left.sent,0)
        self.assertEqual(left.connection.fileno(),-1)

    def test_handshake_protocol_rank_and_byteorder_mismatch(self):
        import sys
        endian=0 if sys.byteorder=='little' else 1
        canonical=[MAGIC,1,endian,VERSION,4096,b'a'*32]
        for index,value in ((0,b'BADMAGIC'),(1,0),(2,1-endian),(3,VERSION+1),(4,0),(4,2**63)):
            left,right=self.pair()
            header=canonical.copy();header[index]=value
            right.connection.sendall(HELLO.pack(*header))
            with self.assertRaises(ValueError):left.handshake()
            self.assertTrue(left.closed);self.assertFalse(left.ready)
            right.close()

    def test_invalid_configuration_retains_caller_socket_ownership(self):
        connection,peer=socket.socketpair()
        self.addCleanup(connection.close);self.addCleanup(peer.close)
        for options in ({'timeout':float('nan')},{'timeout':float('inf')},{'timeout':True},
                        {'timeout':10**1000},{'max_frame_bytes':0},{'max_frame_bytes':2**63}):
            with self.assertRaises(ValueError):HostTensorChannel(connection,b'a'*32,0,**options)
            self.assertGreaterEqual(connection.fileno(),0)
