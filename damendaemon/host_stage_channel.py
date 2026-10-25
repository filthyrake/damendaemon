"""Owned two-rank CPU stage control over an already-connected trusted stream.

This is lifecycle groundwork, not a model worker or authenticated network service.
The caller completes CPU work before acknowledging it and aborts on execution
failure. Any uncertain result discards the epoch; commands are never replayed.
Peers can briefly differ in local commit position after a disconnect: this is
not a durable distributed transaction or a recovery protocol.
"""
import threading
import torch
from .host_tensor_channel import HostTensorChannel
from .stage_protocol import StageState,Command,HEADER_WORDS,VERSION,PREFILL,DECODE,PREFILL_BATCH,PREFILL_CHUNK,RELEASE,STOP,PING,SWITCH,token_rows

ADMISSION,COMPLETION=1,2
EXECUTION=(PREFILL,DECODE,PREFILL_BATCH,PREFILL_CHUNK)


class HostStageChannel:
    """Own a fresh StageState and transport; retain the existing protocol bounds.

    Handshake binds epoch, context, slot count and vocabulary on top of the
    transport's caller-supplied model/session identity. Control and acknowledgements
    use fixed int64 frames. Each frame has the transport's absolute deadline;
    methods contain a fixed number of frames, not an unbounded retry loop.
    One owner serializes commands and activation frames between begin/completion.
    No model allocation, listener, device fence or GPU transfer happens here.
    """
    def __init__(self,transport,state,*,vocab_size):
        if (not isinstance(transport,HostTensorChannel) or transport.ready or transport.closed or transport.failed
                or transport.max_frame_bytes<8*HEADER_WORDS or not isinstance(state,StageState)
                or state.failed or state.closed or state.pending is not None or state.sequence!=1
                or any(state.lengths) or any(state.generations)
                or type(vocab_size) is not int or not 1<=vocab_size<=2**31):
            raise ValueError('host stage requires a fresh owned transport/state and bounded vocabulary')
        self.transport,self.state,self.vocab_size=transport,state,vocab_size
        self.rank=transport.rank;self.ready=self.closed=False
        self.lock=threading.RLock();self.close_lock=threading.RLock()

    def abort(self):
        # Independent of the operation lock so cancellation interrupts socket I/O.
        with self.close_lock:
            self.closed=True;self.ready=False;self.state.poison()
            self.transport.close()

    def close(self):
        with self.close_lock:
            self.closed=True;self.ready=False
            if not self.state.closed:self.state.poison()
            self.transport.close()

    def _usable(self):
        if self.closed or not self.ready or self.state.failed or self.state.closed:
            raise RuntimeError('host stage epoch is not ready')

    def _exchange(self,values):
        tensor=torch.tensor(values,dtype=torch.int64,device='cpu')
        if self.rank==0:
            self.transport.send(tensor)
            return self.transport.receive(tensor.shape,torch.int64).tolist()
        result=self.transport.receive(tensor.shape,torch.int64).tolist()
        self.transport.send(tensor)
        return result

    def handshake(self):
        with self.lock:
            try:
                with self.close_lock:
                    if self.closed or self.ready:raise RuntimeError('host stage handshake cannot be repeated')
                self.transport.handshake()
                expected=[VERSION,self.state.epoch,self.state.context,self.state.slots,self.vocab_size]
                if self._exchange(expected)!=expected:raise ValueError('host stage session bounds disagree')
                with self.close_lock:
                    if self.closed:raise RuntimeError('host stage closed during handshake')
                    self.ready=True
            except BaseException:self.abort();raise

    def _pending(self,command):
        self._usable()
        if not isinstance(command,Command) or self.state.pending!=command:
            raise ValueError('host stage operation does not own pending work')

    def _ack(self,command,phase):
        expected=[command.epoch,command.sequence,phase,1]
        if self._exchange(expected)!=expected:raise ValueError('host stage acknowledgement disagrees')

    def begin(self,command=None):
        with self.lock:
            try:
                self._usable()
                if self.rank==0:
                    if not isinstance(command,Command):raise ValueError('coordinator must supply a stage command')
                    words=command.encode()
                    with self.close_lock:command=self.state.begin(words)
                    self.transport.send(torch.tensor(words,dtype=torch.int64,device='cpu'))
                else:
                    if command is not None:raise ValueError('worker cannot originate a stage command')
                    words=self.transport.receive((HEADER_WORDS,),torch.int64).tolist()
                    with self.close_lock:command=self.state.begin(words)
                self._ack(command,ADMISSION)
                with self.close_lock:self._pending(command)
                return command
            except BaseException:self.abort();raise

    def _commit(self,command):
        self._ack(command,COMPLETION)
        with self.close_lock:
            self._pending(command)
            self.state.commit(command)

    def finish(self,command,success=True):
        """Acknowledge completed release/ping/stop/switch, then commit ownership.

        A switch round commits model_index + a fresh epoch on success; a rank
        whose local weight copy failed aborts the channel instead, poisoning
        both ranks (there is no partial-weight rollback to publish).
        """
        with self.lock:
            try:
                self._pending(command)
                if command.operation not in (RELEASE,PING,STOP,SWITCH):
                    raise ValueError('execution commands require validated token completion')
                if not success:
                    raise RuntimeError('stage round failed on this rank')
                self._commit(command)
                if command.operation==STOP:self.close()
            except BaseException:self.abort();raise

    def _valid_tokens(self,tokens):
        return not bool(((tokens<0)|(tokens>=self.vocab_size)).any())

    def complete_tokens(self,command,tokens=None):
        """Worker supplies CPU int64 sampled IDs; coordinator receives and checks.

        Both ranks commit only after the result transfer and completion exchange.
        Invalid/sentinel IDs abort the epoch; no successful result is returned.
        """
        with self.lock:
            try:
                self._pending(command)
                if command.operation not in EXECUTION:raise ValueError('stage command has no token result')
                shape=(token_rows(command),)
                if self.rank==1:
                    if (not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.int64 or tokens.device.type!='cpu'
                            or tokens.shape!=shape or not tokens.is_contiguous()
                            or not self._valid_tokens(tokens)):
                        raise ValueError('invalid host stage sampled token result')
                    self.transport.send(tokens)
                else:
                    if tokens is not None:raise ValueError('coordinator cannot supply worker results')
                    tokens=self.transport.receive(shape,torch.int64)
                    if not self._valid_tokens(tokens):
                        raise ValueError('invalid host stage sampled token result')
                self._commit(command)
                return tokens
            except BaseException:self.abort();raise
