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
all_reduce MIN, broadcast hops) so the command/ack/transfer ordering is
testable without CUDA; the runtime transport is NCCL with the same calls.
"""
from __future__ import annotations

import torch


class ChainTransport:
    """The collective surface a chain rank needs, injected for testability."""

    def broadcast(self, tensor, source): raise NotImplementedError
    def all_reduce_min(self, tensor): raise NotImplementedError
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

    def __init__(self, role, transport, state, *, device=None):
        from .stage_protocol import HEADER_WORDS
        self.role = role
        self.transport = transport
        self.state = state
        # The command header belongs to the channel, not the stage state:
        # StageState owns only ownership/cursor bookkeeping. NCCL ranks pass
        # the header's device so the broadcast lands on the rank's GPU.
        self.header = torch.zeros(HEADER_WORDS, dtype=torch.int64, device=device)
        # The MIN-ack buffer belongs to the channel; NCCL ranks keep it on
        # the rank's device next to the header.
        self.ack = torch.zeros(1, dtype=torch.int64, device=device)

    def begin(self, command_words=None):
        """Broadcast the coordinator's command; every rank validates and acks."""
        if command_words is not None and not self.role.is_coordinator:
            raise ValueError('only the coordinator originates commands')
        header = self.header
        if self.role.is_coordinator:
            header.copy_(torch.tensor(command_words, dtype=header.dtype, device=header.device))
        self.transport.broadcast(header, 0)
        words = command_words if self.role.is_coordinator else header.cpu().tolist()
        accepted = None
        error = None
        try:
            accepted = self.state.begin(words)
        except (ValueError, RuntimeError) as failure:
            error = failure
        self.ack.fill_(int(error is None))
        self.transport.all_reduce_min(self.ack)
        if not self.ack.item():
            raise ValueError('stage command rejected before activation transfer') from error
        return accepted

    def exchange_hidden(self, tensor, source):
        """Broadcast-chain hop: a stage's hidden output reaches every rank.

        Cross-node P2P send/recv deadlocks this three-rank ring (gate-4
        cycle 1 finding), so every hop is a broadcast that ALL ranks issue
        in the same order. `tensor` holds the payload on the source rank
        and is overwritten on every other rank.
        """
        self.transport.broadcast(tensor, source)
        self.transport.synchronize()

    def exchange_tokens(self, tensor, source):
        """Head rank's token payload broadcast to every rank (same order)."""
        if source != self.role.head_rank:
            raise ValueError('token exchange broadcasts from the head rank')
        self.transport.broadcast(tensor, source)
        self.transport.synchronize()

    def finish(self, command, *, success=True):
        """All-reduce the completion ack, then commit everywhere or poison.

        The all-reduce happens BEFORE any local failure exit: a worker whose
        execution failed must still participate, or the peers block in the
        reduction until the watchdog kills the group (deadlock). On a failed
        MIN ack no state commits and every rank poisons its epoch.
        """
        if self.state.pending != command:
            raise RuntimeError('stage completion does not own pending work')
        self.transport.synchronize()
        self.ack.fill_(int(success))
        self.transport.all_reduce_min(self.ack)
        acknowledged = self.ack.item() == 1
        if acknowledged:
            self.state.commit(command)
        else:
            self.state.poison()
        if not acknowledged:
            raise RuntimeError('chain stage completion rejected; epoch poisoned')
