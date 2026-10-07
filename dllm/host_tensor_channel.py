"""Bounded CPU tensor frames over an owned, already-connected stream socket.

This is a trusted-link transport primitive, not a remote model worker. Identity
agreement is not authentication or encryption. The owner supplies the connected
socket and model/session digest, and orders sends/receives between peers.
Use host_tls to authenticate and pin a connected socket before constructing this
channel. No GPU transfer, listener, pickle, or distributed state commit is performed.
"""
import ctypes
import socket
import struct
import sys
import threading
import time

import torch

MAGIC=b'DLMHOST1'
VERSION=1
MAX_FRAME_BYTES=64*1024*1024
MAX_SEQUENCE=2**64-1
HELLO=struct.Struct('!8sBBHQ32s')
FRAME=struct.Struct('!8sQBBH4QQ')
DTYPES={1:torch.float32,2:torch.float16,3:torch.bfloat16,4:torch.int64,5:torch.int32,6:torch.uint8}
CODES={value:key for key,value in DTYPES.items()}
WIDTHS={torch.float32:4,torch.float16:2,torch.bfloat16:2,torch.int64:8,torch.int32:4,torch.uint8:1}


class HostTensorChannel:
    def __init__(self,connection,identity,rank,*,max_frame_bytes=MAX_FRAME_BYTES,timeout=30):
        if (not isinstance(connection,socket.socket)
                or connection.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)!=socket.SOCK_STREAM):
            raise ValueError('host tensor channel requires a stream socket')
        if not isinstance(identity,bytes) or len(identity)!=32 or type(rank) is not int or rank not in (0,1):
            raise ValueError('host tensor channel requires a 32-byte identity and rank 0 or 1')
        if type(max_frame_bytes) is not int or not 1<=max_frame_bytes<=MAX_FRAME_BYTES:
            raise ValueError('invalid host tensor frame cap')
        if type(timeout) not in (int,float) or not 0<timeout<=300:
            raise ValueError('invalid host tensor operation timeout')
        # Validate before assuming ownership. The caller retains invalid inputs.
        self.connection=connection
        self.identity,self.rank=identity,rank
        self.max_frame_bytes,self.timeout=max_frame_bytes,timeout
        self.ready=self.failed=self.closed=False
        self.sent=self.received=0
        self.lock=threading.RLock()
        self.close_lock=threading.Lock()

    def close(self):
        # Closing from another thread must interrupt blocked socket I/O rather
        # than wait for the operation lock and its deadline.
        with self.close_lock:
            if self.closed:return
            self.closed=True
            self.ready=False
            try:self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            finally:self.connection.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def _usable(self,*,ready=True):
        if self.closed or self.failed or (ready and not self.ready):
            raise RuntimeError('host tensor channel is not ready')

    def _timeout(self,deadline):
        self._usable(ready=False)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('host tensor operation deadline exceeded')
        self.connection.settimeout(min(remaining,.1))

    def _send(self,data,deadline):
        view=memoryview(data).cast('B')
        try:
            offset=0
            while offset<len(view):
                self._timeout(deadline)
                try:count=self.connection.send(view[offset:])
                except TimeoutError:continue
                if not count:raise EOFError('host tensor peer closed during send')
                offset+=count
        finally:view.release()

    def _read(self,size,deadline):
        data=bytearray(size)
        view=memoryview(data)
        try:
            offset=0
            while offset<size:
                self._timeout(deadline)
                try:count=self.connection.recv_into(view[offset:])
                except TimeoutError:continue
                if not count:raise EOFError('host tensor peer closed during receive')
                offset+=count
        finally:view.release()
        return data

    def _fail(self):
        self.failed=True
        self.close()

    def handshake(self):
        with self.lock:
            self._usable(ready=False)
            if self.ready:raise RuntimeError('host tensor handshake already completed')
            deadline=time.monotonic()+self.timeout
            endian=0 if sys.byteorder=='little' else 1
            try:
                self._send(HELLO.pack(MAGIC,self.rank,endian,VERSION,self.max_frame_bytes,self.identity),deadline)
                magic,rank,peer_endian,version,limit,identity=HELLO.unpack(self._read(HELLO.size,deadline))
                if (magic!=MAGIC or rank!=1-self.rank or peer_endian!=endian or version!=VERSION
                        or identity!=self.identity or not 1<=limit<=MAX_FRAME_BYTES):
                    raise ValueError('host tensor peer identity or protocol mismatch')
                self.max_frame_bytes=min(self.max_frame_bytes,limit)
                with self.close_lock:
                    self._usable(ready=False)
                    self.ready=True
            except BaseException:
                self._fail()
                raise

    def _layout(self,shape,dtype):
        if (not isinstance(dtype,torch.dtype) or dtype not in CODES or not isinstance(shape,(tuple,list,torch.Size)) or not 1<=len(shape)<=4
                or any(type(value) is not int or value<1 for value in shape)):
            raise ValueError('unsupported host tensor layout')
        size=WIDTHS[dtype]
        for value in shape:
            if value>self.max_frame_bytes//size:raise ValueError('host tensor exceeds frame cap')
            size*=value
        return tuple(shape),size

    def send(self,tensor):
        """Send a contiguous CPU tensor; caller must not mutate it until return."""
        with self.lock:
            self._usable()
            if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu'
                    or tensor.layout!=torch.strided or not tensor.is_contiguous()
                    or tensor.is_conj() or tensor.is_neg()):
                raise ValueError('host tensor send requires contiguous ordinary CPU storage')
            shape,size=self._layout(tensor.shape,tensor.dtype)
            if self.sent>=MAX_SEQUENCE:raise RuntimeError('host tensor send sequence exhausted')
            sequence=self.sent+1
            header=FRAME.pack(MAGIC,sequence,CODES[tensor.dtype],len(shape),0,
                              *(shape+(0,)*(4-len(shape))),size)
            deadline=time.monotonic()+self.timeout
            try:
                self._send(header,deadline)
                # Keep tensor alive while the socket reads its bytes. No NumPy,
                # tensor serialization, or extra contiguous copy is involved.
                data=(ctypes.c_ubyte*size).from_address(tensor.data_ptr())
                self._send(data,deadline)
                with self.close_lock:
                    self._usable()
                    self.sent=sequence
            except BaseException:
                self._fail()
                raise

    def receive(self,shape,dtype):
        """Require the expected layout before allocating any payload storage."""
        with self.lock:
            self._usable()
            shape,size=self._layout(shape,dtype)
            if self.received>=MAX_SEQUENCE:raise RuntimeError('host tensor receive sequence exhausted')
            deadline=time.monotonic()+self.timeout
            try:
                values=FRAME.unpack(self._read(FRAME.size,deadline))
                magic,sequence,code,rank,flags=values[:5]
                dimensions=values[5:9]
                length=values[9]
                if (magic!=MAGIC or sequence!=self.received+1 or code!=CODES[dtype]
                        or rank!=len(shape) or flags!=0 or dimensions!=shape+(0,)*(4-len(shape))
                        or length!=size):
                    raise ValueError('host tensor frame metadata mismatch')
                data=self._read(size,deadline)
                # frombuffer retains its bytearray owner; every frame has its own
                # writable storage and cannot alias later received frames.
                result=torch.frombuffer(data,dtype=dtype).reshape(shape)
                with self.close_lock:
                    self._usable()
                    self.received=sequence
                return result
            except BaseException:
                self._fail()
                raise
