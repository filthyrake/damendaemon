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


def bounded_exchange(exchange, deadline, *, poll_interval=0.05,
                     clock=time.monotonic, sleep=time.sleep):
    """Run exchange() under an absolute deadline, off the caller's thread.

    Live-measured (2026-10-28, cross-host A2+3090 cell): when a peer dies,
    the async collective POST itself can block inside the NCCL comm locks
    (the watchdog stack showed the run thread inside the async_op=True
    broadcast call), so a poll-only deadline never arms and the 30s group
    watchdog terminates the coordinator. The exchange runs in a daemon
    thread; if it does not RETURN within `deadline` the caller gets a
    transport-marked PeerDeadlineExceeded and can abort the group, while
    the stuck helper thread is abandoned (disclosed; the abort normally
    releases it). A healthy exchange returns in microseconds.
    """
    if not callable(exchange):
        raise ValueError('bounded exchange needs a callable')
    if not 0 < deadline <= 600 or not 0 < poll_interval <= 10:
        raise ValueError('invalid poll deadline or interval bounds')
    import threading
    outcome = {}
    done = threading.Event()

    def runner():
        try:
            outcome['result'] = exchange()
        except BaseException as error:
            outcome['error'] = error
        finally:
            done.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    end = clock() + deadline
    while not done.is_set():
        if clock() >= end:
            raise PeerDeadlineExceeded(
                f'peer heartbeat timeout: the polled exchange did not '
                f'return within {deadline:g}s (async collective post '
                'blocked; connection/transport deadline exceeded)')
        sleep(poll_interval)
    if 'error' in outcome:
        raise outcome['error']
    return outcome.get('result')


def abort_process_group_safely():
    """Best-effort abort of this process's process groups.

    Called only after a polled exchange has failed (deadline or collective
    error): the group is unusable, and destroying it directly can hang in
    NCCL teardown with in-flight or errored work — the abort clears the
    registry so a later rejoin builds a fresh group in this process
    (measured live 2026-10-28: without it, the supervisor's close path
    froze the whole coordinator inside destroy_process_group, GIL held).
    Returns True when the abort ran, False when unavailable or it raised.
    """
    if _abort_process_group is None:
        return False
    try:
        _abort_process_group()
        return True
    except Exception:
        return False


def abort_dead_group_bounded(*, bound=5.0):
    """Bounded abandonment of a dead group's abort.

    Measured live (2026-10-28 cross-host cell, generation 2): the abort
    itself can block in NCCL teardown locks while the run thread waits on
    it — the supervisor state stayed 'serving' (lying /health) until the
    30s group watchdog terminated the process. The abort runs in a daemon
    thread with a `bound`-second join: the caller proceeds to raise
    immediately either way. When the abort hangs it is abandoned (leaked
    thread, disclosed) and the coordinator stays on the watchdog fuse —
    bounded recovery with honest states; the outer process deadline
    remains mandatory for that residual.
    """
    import threading
    finished = []

    def runner():
        abort_process_group_safely()
        finished.append(True)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(bound)
    return bool(finished)


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

        The ENTIRE exchange runs through `bounded_exchange`: when a peer
        dies, the async collective post itself can block in NCCL comm
        locks (measured live — the watchdog stack showed the run thread
        inside the async_op=True broadcast post, so a poll-only deadline
        never armed and the 30s group watchdog terminated the coordinator,
        the #637 signature). With the bound, a peer that stops completing
        heartbeats surfaces as a catchable PeerDeadlineExceeded at
        `deadline` (keep it below the group timeout so this side always
        classifies first), the state is poisoned and the dead group's
        in-flight work aborted (clearing the process-group registry so a
        rejoin builds a fresh group in this process). The worker rank keeps
        its blocking exchange — an async post on this side completes it
        fine. Mid-work peer death keeps the existing disclosed
        process-fatal behavior; only the idle heartbeat is polled.
        """
        if self.rank != 0:
            raise ValueError('only coordinator sends heartbeats')
        try:
            bounded_exchange(
                lambda: self._exchange_ping(deadline, poll_interval=poll_interval,
                                            clock=clock, sleep=sleep,
                                            dist_module=dist_module),
                deadline, poll_interval=poll_interval, clock=clock, sleep=sleep)
        except BaseException:
            # Any failure (deadline, blocked post, failed collective) is a
            # dead group: poison, then a BOUNDED abort (the abort itself
            # was measured to block in teardown locks — a synchronous wait
            # would leave the supervisor lying 'serving' until the group
            # watchdog terminates the process), then raise.
            self.state.poison()
            abort_dead_group_bounded()
            raise

    def _exchange_ping(self, deadline, *, poll_interval, clock, sleep,
                       dist_module):
        """The PING begin+finish sequence on async collectives, polled."""
        distributed = dist if dist_module is None else dist_module
        command = Command(self.state.epoch, self.state.sequence, PING, 0, ())
        words = command.encode()
        works = []
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
