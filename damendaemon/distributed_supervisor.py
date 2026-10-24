"""Operator-triggered supervision for the two-rank distributed service.

Detection reuses the existing heartbeat/ack deadline machinery: the stage
transport frames already carry absolute deadlines, so a dead peer surfaces
as a failed exchange inside the runtime's run loop, which poisons the
session honestly (disclosed at the HTTP boundary as before). The supervisor
adds what the operator had to do by hand: classify the failure, keep the
HTTP listener up in a failed state, and perform an explicit, bounded rejoin
on request — a fresh worker process and pair re-negotiate identity and
epoch, and the service resumes accepting requests with empty slots.

Honest boundaries, unchanged from the issue: no in-flight request replay
across a rejoin (admitted requests of a dead generation fail with their
generation and restart from their admitted state if resubmitted); no
automatic restart policy; after the generation bound is exhausted the
supervisor enters a terminal state rather than looping.
"""
import os
import threading


PEER_DEAD_MARKERS = ('timeout', 'nccl', 'connection', 'socket', 'transport')


def classify_failure(error):
    """Best-effort dead-peer classification from the poisoned session error.

    Returns 'peer-dead' when the failure chain names a transport deadline,
    connection or NCCL/RCCL error; 'failed' otherwise (still rejoinable by
    an explicit operator request — the classification is diagnostics, not
    admission).
    """
    seen = 0
    current = error
    while current is not None and seen < 8:
        text = str(current).lower()
        if (isinstance(current, (TimeoutError, ConnectionError, OSError))
                or any(marker in text for marker in PEER_DEAD_MARKERS)):
            return 'peer-dead'
        current = current.__cause__ or current.__context__
        seen += 1
    return 'failed'


class SupervisorRuntime:
    """Handler-facing proxy over the current generation's runtime."""

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.rejoin_lock = threading.Lock()

    @property
    def inner(self):
        return self.supervisor.current_runtime

    @property
    def closed(self):
        """Event-like truth for the HTTP health path."""
        class _Closed:
            def __init__(self, supervisor):self.supervisor = supervisor
            def is_set(self):return self.supervisor.state in ('closed', 'terminal')
        return _Closed(self.supervisor)

    def health(self):
        """Distributed-serve health extension: supervisor state and reason."""
        state = self.supervisor.state
        if state == 'serving':
            return 200, dict(status='ok', generation=self.supervisor.generation)
        return 503, dict(status=state, generation=self.supervisor.generation,
                         reason=self.supervisor.reason,
                         rejoin='POST /v1/distributed/rejoin' if self.supervisor.rejoinable() else None)

    def rejoin(self):
        """Operator-triggered rejoin; bounded and serialized."""
        with self.rejoin_lock:
            return self.supervisor.rejoin()

    def manage(self, action, **options):
        if action == 'status':
            base = self.inner.manage('status') if self.inner is not None else {}
            base.update(supervisor=dict(state=self.supervisor.state,
                                        generation=self.supervisor.generation,
                                        reason=self.supervisor.reason))
            return base
        return self.inner.manage(action, **options)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class DistributedSupervisor:
    """Own generation lifetimes: (worker process, pair, runtime) tuples.

    `runtime_builder(port, run_id, worker)` must return (pair, runtime) for
    the coordinator rank and is fully injected so CPU-bus failure injection
    needs no GPU: the default builder spawns the real rank-1 worker subprocess
    and constructs StagePair/DistributedRuntime in this process.
    """

    def __init__(self, runtime_builder, *, max_generations=3, monitor_interval=0.5):
        if not callable(runtime_builder):raise ValueError('supervisor needs a runtime builder')
        if not 1 <= max_generations <= 16:raise ValueError('generation bound must be 1..16')
        self.builder = runtime_builder
        self.max_generations = max_generations
        self.monitor_interval = monitor_interval
        self.generation = 0
        self.state = 'idle'
        self.reason = None
        self.current_runtime = None
        self.current_pair = None
        self.run_thread = None
        self.stopping = False
        self.lock = threading.RLock()
        self.runtime_proxy = SupervisorRuntime(self)

    def rejoinable(self):
        with self.lock:
            return self.state in ('peer-dead', 'failed') and self.generation < self.max_generations

    def start(self):
        with self.lock:
            if self.state != 'idle':raise RuntimeError('supervisor already started')
            self._build('initial generation')

    def _build(self, reason):
        with self.lock:
            if self.stopping:raise RuntimeError('supervisor is shutting down')
            self.generation += 1
        try:
            pair, runtime = self.builder(self.generation)
        except BaseException as error:
            self.current_pair = self.current_runtime = None
            self.state = 'terminal' if self.generation >= self.max_generations else 'failed'
            self.reason = f'generation {self.generation} failed to build: {error}'
            raise
        self.current_pair, self.current_runtime = pair, runtime
        if self.stopping:
            for closer in (runtime.close, pair.stop, pair.close):
                try:closer()
                except Exception:pass
            raise RuntimeError('supervisor is shutting down')
        self.state, self.reason = 'serving', None
        self.run_thread = threading.Thread(target=self._supervised_run,
                                           args=(self.generation, pair, runtime), daemon=True)
        self.run_thread.start()

    def _supervised_run(self, generation, pair, runtime):
        error = None
        try:
            runtime.run(idle_timeout=0)
        except BaseException as raised:
            error = raised
        with self.lock:
            # A stale generation's death must not touch its successor.
            if self.generation != generation or self.state != 'serving':
                return
            if error is None:
                self.state, self.reason = 'closed', 'runtime run loop exited cleanly'
                return
            self.state = classify_failure(error)
            self.reason = f'generation {generation}: {type(error).__name__}: {error}'
            for closer in (runtime.close, pair.stop, pair.close):
                try:closer()
                except Exception:pass
            self.current_runtime = self.current_pair = None

    def rejoin(self):
        """One operator-triggered rejoin; asynchronous, returns (initiated, detail)."""
        with self.lock:
            if self.state == 'rejoining':
                return False, 'rejoin already in progress'
            if self.state == 'serving':
                return False, 'current generation is serving; rejoin is for dead generations'
            if self.state == 'idle':
                return False, 'supervisor never started'
            if self.generation >= self.max_generations:
                self.state = 'terminal'
                return False, (f'generation bound {self.max_generations} exhausted; '
                               'restart the service (terminal state, no automatic retry)')
            self.state = 'rejoining'
            self.reason = None
            threading.Thread(target=self._rejoin_build, daemon=True).start()
            return True, (f'rejoining to generation {self.generation + 1}; slots will be '
                          'empty; admitted requests of the dead generation are not replayed')

    def _rejoin_build(self):
        try:
            self._build('operator rejoin')
        except BaseException:
            pass

    def close(self):
        with self.lock:
            runtime, pair = self.current_runtime, self.current_pair
            if runtime is not None:
                try:runtime.close()
                except Exception:pass
            if self.run_thread is not None:
                self.run_thread.join(timeout=10)
            self.stopping = True
            self.state = 'closed' if self.state != 'terminal' else self.state
            if pair is not None:
                try:pair.stop()
                except Exception:pass
                try:pair.close()
                except Exception:pass
                self.current_runtime = self.current_pair = None
