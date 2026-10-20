"""Experimental three-rank hybrid chain control channel (#536 gate 4, cycle 1).

Extends the proven two-rank stage protocol shape (stage_transport.StageChannel)
to a chained topology: rank 0 = coordinator + stage 0, middle ranks own one
stage each, the last rank additionally owns the output head and returns
sampled tokens to rank 0. Rank r's forward peer is r+1; the token return is
head_rank -> 0.

Control-plane only: command broadcast (rank 0 -> all), MIN all-reduce
acknowledgements, and point-to-point hidden/token transfer. Model loading,
identity agreement and HTTP serving stay with the caller (the two-rank
StagePair machinery remains the reference; this module does not replace it).

A CPU test transport implements the same collective surface (broadcast,
all_reduce MIN, batch P2P isend/irecv) so the command/ack/transfer ordering is
testable without CUDA; the runtime transport is NCCL with the same calls.
"""
from __future__ import annotations

import torch


class ChainTransport:
    """The collective surface a chain rank needs, injected for testability."""

    def broadcast(self, tensor, source): raise NotImplementedError
    def all_reduce_min(self, tensor): raise NotImplementedError
    def send(self, tensor, peer): raise NotImplementedError
    def recv(self, tensor, peer): raise NotImplementedError
    def synchronize(self): raise NotImplementedError


class NcclChainTransport(ChainTransport):
    """Runtime transport; requires an initialized NCCL process group."""

    def __init__(self):
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_backend() != 'nccl':
            raise ValueError('NcclChainTransport requires an initialized NCCL group')
        self._dist = dist

    def broadcast(self, tensor, source):
        self._dist.broadcast(tensor, source)

    def all_reduce_min(self, tensor):
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.MIN)

    def send(self, tensor, peer):
        for work in self._dist.batch_isend_irecv([self._dist.P2POp(self._dist.isend, tensor, peer)]):
            work.wait()

    def recv(self, tensor, peer):
        for work in self._dist.batch_isend_irecv([self._dist.P2POp(self._dist.irecv, tensor, peer)]):
            work.wait()

    def synchronize(self):
        torch.cuda.synchronize()


class ChainRole:
    """Validated rank geometry for one chain member."""

    def __init__(self, rank, world_size):
        if (type(rank) is not int or type(world_size) is not int
                or not 2 <= world_size <= 16 or not 0 <= rank < world_size):
            raise ValueError('chain roles require 2..16 ranks and a valid rank')
        self.rank = rank
        self.world_size = world_size
        self.head_rank = world_size - 1
        self.next_rank = rank + 1 if rank + 1 < world_size else None
        self.owns_head = rank == self.head_rank
        self.is_coordinator = rank == 0

    def forward_peer(self):
        """The peer this rank sends hidden states to (None on the head rank)."""
        if self.next_rank is None:
            raise ValueError('the head rank has no forward peer')
        return self.next_rank


class ChainChannel:
    """Command broadcast + ack + forward/token transfer for a chain member."""

    def __init__(self, role, transport, state):
        self.role = role
        self.transport = transport
        self.state = state

    def begin(self, command_words=None):
        """Broadcast the coordinator's command; every rank validates and acks."""
        if command_words is not None and not self.role.is_coordinator:
            raise ValueError('only the coordinator originates commands')
        header = self.state.header
        if self.role.is_coordinator:
            header.copy_(torch.tensor(command_words, dtype=header.dtype, device=header.device))
        self.transport.broadcast(header, 0)
        accepted = None
        error = None
        try:
            accepted = self.state.begin(None if self.role.is_coordinator else header.cpu().tolist())
        except (ValueError, RuntimeError) as failure:
            error = failure
        self.transport.ack.fill_(int(error is None))
        self.transport.all_reduce_min(self.transport.ack)
        if not self.transport.ack.item():
            raise ValueError('stage command rejected before activation transfer') from error
        return accepted

    def forward_hidden(self, hidden):
        """Send this rank's stage output to the next rank (blocking, synced)."""
        peer = self.role.forward_peer()
        self.transport.send(hidden, peer)
        self.transport.synchronize()

    def receive_hidden(self, hidden):
        """Receive the previous rank's stage output into this rank's buffer."""
        if self.role.rank == 0:
            raise ValueError('the coordinator receives no hidden state')
        self.transport.recv(hidden, self.role.rank - 1)
        self.transport.synchronize()

    def return_tokens(self, tokens):
        """Head rank sends sampled int64 tokens to the coordinator (fenced)."""
        if not self.role.owns_head:
            raise ValueError('only the head rank returns tokens')
        self.transport.send(tokens, 0)
        self.transport.synchronize()

    def receive_tokens(self, tokens):
        """Coordinator receives the head rank's tokens; also the completion fence."""
        if not self.role.is_coordinator:
            raise ValueError('only the coordinator receives tokens')
        self.transport.recv(tokens, self.role.head_rank)
        self.transport.synchronize()

    def finish(self, command, *, success=True):
        if self.state.pending != command or self.state.failed:
            raise RuntimeError('stage completion does not own pending work')
        self.transport.synchronize()
        self.transport.ack.fill_(int(success))
        self.transport.all_reduce_min(self.transport.ack)
        if self.transport.ack.item() != 1:
            raise RuntimeError('stage completion acknowledgement failed')
        self.state.commit(command)
