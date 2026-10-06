"""Experimental NCCL control channel for a trusted two-worker process group.

The owner provides group deadlines and tears down both workers on failure.
Model identity must be checked before constructing this channel. This module
has no model loading, discovery, authentication, or HTTP service.
"""
import torch
import torch.distributed as dist
from .stage_protocol import HEADER_WORDS


class StageChannel:
    def __init__(self, state, rank):
        if not dist.is_initialized() or dist.get_world_size() != 2 or rank != dist.get_rank():
            raise ValueError('stage channel requires a matching two-worker group')
        if dist.get_backend() != 'nccl':
            raise ValueError('stage channel requires NCCL')
        self.state = state
        self.rank = rank
        self.header = torch.empty(HEADER_WORDS, dtype=torch.int64, device='cuda')
        self.acknowledgement = torch.empty(1, dtype=torch.int64, device='cuda')

    def begin(self, command=None):
        try:
            words = None
            if self.rank == 0:
                if command is None:raise ValueError('coordinator command required')
                words = command.encode()
                self.header.copy_(torch.tensor(words, dtype=torch.int64, device='cuda'))
            elif command is not None:
                raise ValueError('worker cannot originate commands')
            dist.broadcast(self.header, 0)
            error = None
            try:accepted = self.state.begin(self._received_words(words))
            except (ValueError, RuntimeError) as failure:error = failure
            self.acknowledgement.fill_(int(error is None))
            dist.all_reduce(self.acknowledgement, op=dist.ReduceOp.MIN)
            if not self.acknowledgement.item():
                raise ValueError('stage command rejected before activation transfer') from error
            return accepted
        except BaseException:
            self.state.poison()
            raise

    def _received_words(self, sent_words):
        # The broadcast source already owns the exact encoded record. Only the
        # receiving worker needs a device-to-host copy. Both validate admission
        # and observe the collective result before any activation transfer.
        return sent_words if self.rank == 0 else self.header.cpu().tolist()

    def finish(self, command):
        try:
            if self.state.pending != command or self.state.failed:
                raise RuntimeError('stage completion does not own pending work')
            # All work must complete before the peer may reuse a released slot.
            torch.cuda.synchronize()
            self.acknowledgement.fill_(1)
            dist.all_reduce(self.acknowledgement, op=dist.ReduceOp.MIN)
            # wait() is not a host completion fence for asynchronous NCCL work.
            # Reading the result also catches a failed collective before commit.
            if self.acknowledgement.item() != 1:
                raise RuntimeError('stage completion acknowledgement failed')
            self.state.commit(command)
        except BaseException:
            self.state.poison()
            raise

    def complete_tokens(self, command, tokens):
        """Final-stage token transfer is also the execution completion fence.

Rank one supplies sampled int64 token IDs; rank zero supplies a receive buffer.
A reserved negative result can report an output-stage execution exception;
the owning StagePair rejects it and poisons the session. No full vocabulary
crosses the boundary. Send/receive and CUDA completion must
succeed before either side commits ownership. Release/stop use finish().
"""
        from .stage_protocol import PREFILL, DECODE, PREFILL_BATCH, token_rows
        try:
            rows = token_rows(command)
            if (self.state.pending != command or command.operation not in (PREFILL, DECODE, PREFILL_BATCH)
                    or tokens.dtype != torch.int64 or tokens.shape != (rows,)
                    or tokens.device.type != 'cuda'
                    or tokens.device.index != torch.cuda.current_device()):
                raise ValueError('invalid stage token completion')
            operation = dist.P2POp(dist.isend if self.rank == 1 else dist.irecv, tokens, 1 - self.rank)
            for work in dist.batch_isend_irecv([operation]):work.wait()
            torch.cuda.synchronize()
            self.state.commit(command)
            return tokens
        except BaseException:
            self.state.poison()
            raise
