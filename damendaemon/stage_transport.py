"""Experimental NCCL control channel for a trusted two-worker process group.

The owner provides group deadlines and tears down both workers on failure.
Model identity must be checked before constructing this channel. This module
has no model loading, discovery, authentication, or HTTP service.
"""
import time

import torch
import torch.distributed as dist
from .stage_protocol import HEADER_WORDS, PING, Command

try:
    # Underscore names are skipped by the star-import onto torch.distributed,
    # so the experimental NCCL group abort is only reachable here.
    from torch.distributed.distributed_c10d import _abort_process_group
except ImportError:  # pragma: no cover - older torch without the recovery API
    _abort_process_group = None


class PeerDeadlineExceeded(RuntimeError):
    """A polled control exchange did not complete within its deadline.

    Raised by the supervised idle heartbeat when a peer stops completing
    collectives: the exchange is abandoned without blocking on the dead
    group, the group's in-flight work is aborted, and the supervisor gets
    a catchable classification signal instead of the process-fatal NCCL
    watchdog timeout (the #637 finding). The message deliberately names
    a transport timeout so `classify_failure` maps it to 'peer-dead'.
    """


def poll_works(works, deadline, *, poll_interval=0.05, clock=time.monotonic,
               sleep=time.sleep):
    """Wait for async collective handles without blocking past `deadline`.

    Pure poll loop: returns when every handle reports completion, raises
    PeerDeadlineExceeded when the absolute deadline passes first. Clock,
    sleep and the handles are injected so CPU-bus tests drive the exact
    semantics without a process group.
    """
    if not 0 < deadline <= 600 or not 0 < poll_interval <= 10:
        raise ValueError('invalid poll deadline or interval bounds')
    if not works or any(not hasattr(work, 'is_completed') for work in works):
        raise ValueError('poll requires async work handles')
    end = clock() + deadline
    while True:
        if all(work.is_completed() for work in works):
            return
        if clock() >= end:
            raise PeerDeadlineExceeded(
                f'peer heartbeat timeout: polled NCCL exchange did not '
                f'complete within {deadline:g}s; peer is not completing '
                'collectives (connection/transport deadline exceeded)')
        sleep(poll_interval)


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

    def poll_ping(self, deadline, *, poll_interval=0.05, clock=time.monotonic,
                  sleep=time.sleep, dist_module=None):
        """Idle keep-alive exchange with a polled, non-blocking deadline.

        Replicates the blocking PING path (begin + finish: header broadcast,
        acknowledgement all_reduce, completion all_reduce) using async
        collectives polled with `poll_works`, so a dead peer surfaces as a
        catchable PeerDeadlineExceeded at `deadline` instead of the
        process-fatal blocking wait at the group timeout (the #637 finding:
        the NCCL watchdog took the whole coordinator down before the
        supervisor's rejoin path could engage). The worker rank keeps its
        blocking exchange — an async post on this side completes it fine.

        On deadline: poison the state, abort the dead group's in-flight
        work via torch's experimental `_abort_process_group` when present
        (abort clears the process-group registry, letting a rejoin build a
        fresh group in this process; without an abort, destroy/re-init can
        hang on the abandoned collective), then raise. `deadline` must sit
        below the process-group timeout so this side always classifies
        first.
        """
        if self.rank != 0:
            raise ValueError('only coordinator sends heartbeats')
        distributed = dist if dist_module is None else dist_module
        command = Command(self.state.epoch, self.state.sequence, PING, 0, ())
        words = command.encode()
        works = []
        try:
            self.header.copy_(
                torch.tensor(words, dtype=torch.int64, device='cuda'))
            works.append(distributed.broadcast(self.header, 0, async_op=True))
            poll_works(works, deadline, poll_interval=poll_interval,
                      clock=clock, sleep=sleep)
            accepted = self.state.begin(words)
            self.acknowledgement.fill_(1)
            works.append(
                distributed.all_reduce(self.acknowledgement,
                                       op=distributed.ReduceOp.MIN,
                                       async_op=True))
            poll_works(works, deadline, poll_interval=poll_interval,
                       clock=clock, sleep=sleep)
            torch.cuda.synchronize()
            if self.acknowledgement.item() != 1:
                raise RuntimeError('polled heartbeat acknowledgement failed')
            self.state.commit(accepted)
            self.acknowledgement.fill_(1)
            works.append(
                distributed.all_reduce(self.acknowledgement,
                                       op=distributed.ReduceOp.MIN,
                                       async_op=True))
            poll_works(works, deadline, poll_interval=poll_interval,
                       clock=clock, sleep=sleep)
            if self.acknowledgement.item() != 1:
                raise RuntimeError('polled heartbeat completion failed')
        except PeerDeadlineExceeded:
            self.state.poison()
            if _abort_process_group is not None:
                try:
                    _abort_process_group()
                except Exception:
                    pass
            raise
        except BaseException:
            self.state.poison()
            raise

    def finish(self, command, *, success=True):
        try:
            if self.state.pending != command or self.state.failed:
                raise RuntimeError('stage completion does not own pending work')
            # All work must complete before the peer may reuse a released slot.
            torch.cuda.synchronize()
            self.acknowledgement.fill_(int(success))
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
