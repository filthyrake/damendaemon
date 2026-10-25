"""Bounded listener daemon exposing this host's stage to a remote peer (#603).

No silent enrollment: nothing binds until a WorkerDaemon is explicitly
constructed, and construction fails unless the configured model loads and
fingerprints cleanly. The daemon speaks the EXISTING qualified two-rank
protocol — HostTensorChannel frames + HostStageChannel commands through a
HostHybridWorker — over the existing mutual-TLS machinery with certificate
pins and the damendaemon-host/1 ALPN; no second protocol is invented.

Session lifecycle: one exclusive client session at a time. A client that
arrives while the session slot is held is refused at accept time, or —
with an explicit bounded `session_queue` — parked FIFO until the slot
frees or the queue wait bound expires. Each session gets a fresh stage
through the factory and any protocol, execution or handshake failure
poisons that session only, returning the daemon to idle. Client
disconnect and idle timeouts end the session through the channel's own
transport discipline. An optional pre-agreed model table enables live
`switch_model` on idle sessions (see the #603 section of
docs/multi-gpu.md). No multi-tenant scheduling, no automatic restart, no
serving admission and no throughput claims; the caller supplies the bind
address, and loopback is the safe default for testing.
"""
import collections
import socket
import threading
import time

from .host_hybrid_worker import HostHybridWorker
from .host_stage_channel import HostStageChannel
from .host_tensor_channel import HostTensorChannel
from .host_tls import ALPN, certificate_pin, server_context, wrap_peer
from .stage_protocol import StageState
from .stage_rebind import detached_model


def _resident_model(stage):
    """Detach a validation stage's full source model as a switch source.

    The stage's own .model is the layer shard; the switch table needs the
    complete prepared weights (and a fingerprint matching the peer's full
    model), which the stage retains as .source_model. close() only clears
    the shard's view, so the detached copy stays valid after validation.
    """
    return detached_model(getattr(stage, 'source_model', stage.model))


class WorkerDaemon:
    """Own a listener plus a validated model after construction.

    Serve in one owned thread via serve_forever(); close() interrupts the
    listener and the active session (the channel's cancellation discipline
    unwinds blocked transport) and waits at most close_timeout seconds.
    Failed cleanup stays retryable: the owner retains the exception.

    The existing stage protocol closes a stage on STOP, so no stage can
    outlive its session; therefore construction loads the model ONCE to
    validate and fingerprint it and then releases those weights, and every
    session rebuilds a fresh stage through the factory. The validation
    copy never serves a request.
    """

    def __init__(self, address, stage_factory, *, identity, tls_dir, client_pin,
                 max_frame_bytes=64 * 1024 * 1024,
                 timeout=30, close_timeout=10, session_queue=0,
                 additional_models=()):
        if (not isinstance(address, tuple) or len(address) != 2
                or not isinstance(address[0], str) or type(address[1]) is not int
                or not 0 <= address[1] <= 65535
                or not callable(stage_factory)
                or not isinstance(identity, bytes) or len(identity) != 32
                or type(max_frame_bytes) is not int or not 1 <= max_frame_bytes <= 2 ** 31
                or type(timeout) is not int or not 0 < timeout <= 30
                or type(close_timeout) is not int or not 0 < close_timeout <= 30
                or type(session_queue) is not int or not 0 <= session_queue <= 16):
            raise ValueError('invalid worker daemon configuration')
        # Load and fingerprint the model BEFORE binding any socket: a broken
        # checkpoint must fail construction, not a client session. The
        # validation copy is released immediately; it never serves.
        self.stage_factory = stage_factory
        self.max_frame_bytes = max_frame_bytes
        stage = stage_factory()
        try:
            if (stage.closed or getattr(stage, 'failed', False)
                    or not stage.last or stage.first
                    or any(stage.lengths)):
                raise ValueError('worker daemon requires a fresh final-range stage')
            self.context = stage.context
            self.slots = stage.max_batch
            self.vocab_size = stage.c['vocab_size']
            self.hidden = stage.c['hidden_size']
            self.layer_end = stage.layer_end
        finally:
            stage.close()
        # The session identity is the FULL model's fingerprint, exactly as
        # the qualified cell pattern supplies it to both ranks; the stage
        # shard's own integrity is pinned by the worker contract handshake.
        self.identity = identity
        # Optional pre-agreed switch table (#603): the primary model plus at
        # most one additional alias. Each additional factory is validated the
        # same way (fresh final-range stage) BEFORE binding, and its prepared
        # model stays resident for the daemon's lifetime — switching trades
        # residency for latency, it does not stream weights from disk. With
        # no table, nothing is retained: the validation stage is still
        # released, and no source clone doubles the model in memory.
        if (not isinstance(additional_models, (list, tuple))
                or 1 + len(additional_models) > 2
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       or not isinstance(pair[0], str) or not 1 <= len(pair[0]) <= 128
                       or not callable(pair[1]) for pair in additional_models)
                or len({alias for alias, _ in additional_models} | {'model'})
                != 1 + len(additional_models)):
            raise ValueError('worker daemon model table requires unique aliases '
                             'and stage factories (at most two models)')
        self.models = None
        if additional_models:
            self.models = {'model': _resident_model(stage)}
            for alias, factory in additional_models:
                extra = factory()
                try:
                    if (extra.closed or getattr(extra, 'failed', False)
                            or not extra.last or extra.first
                            or any(extra.lengths)):
                        raise ValueError('worker daemon requires a fresh final-range stage')
                    if extra.model.config != self.models['model'].config:
                        raise ValueError('worker daemon model switching requires '
                                         'identical decoder configurations')
                    self.models[alias] = _resident_model(extra)
                finally:
                    extra.close()
        # TLS: the existing server context + the single admitted client pin.
        self.context_tls = server_context(ca_file=str(tls_dir / 'ca.pem'),
                                          cert_file=str(tls_dir / 'server.pem'),
                                          key_file=str(tls_dir / 'server.key'),
                                          alpn=ALPN)
        self.client_pin = certificate_pin(client_pin)
        self.timeout = timeout
        self.close_timeout = close_timeout
        # One exclusive session: the gate is a single-slot semaphore. A busy
        # daemon refuses a new client AT ACCEPT TIME when session_queue is 0
        # (the default); with a bounded queue (1..16), up to that many
        # waiting clients hold their TLS connection in FIFO order until the
        # session slot frees or the queue wait bound (the daemon timeout)
        # expires, whichever comes first. Queue waiters never execute: they
        # only hold a raw connection until the pump serves them.
        self.stopping = threading.Event()
        self.gate = threading.BoundedSemaphore(1)
        self.session_queue = session_queue
        self.queue = collections.deque()
        self.pumping = False
        self.table_size = 1 + len(additional_models)
        self.workers = {}
        self.lock = threading.Lock()
        self.acceptor = None
        self.closed = False
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(address)
            listener.listen(4)
            listener.settimeout(.1)
        except BaseException:
            listener.close()
            raise
        self.listener = listener
        self.address = listener.getsockname()[:2]

    # -- lifecycle ---------------------------------------------------------

    def serve_forever(self):
        with self.lock:
            if self.acceptor is not None or self.stopping.is_set():
                raise RuntimeError('worker daemon already serving or stopping')
            self.acceptor = threading.current_thread()
        try:
            while not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self.stopping.is_set():
                        return
                    raise
                if not self.gate.acquire(blocking=False):
                    # Exclusive session policy. session_queue == 0 refuses
                    # the client now; a bounded queue parks the connection
                    # FIFO until the slot frees, the queue bound is hit
                    # (refuse the newest), or the daemon stops.
                    if not self._enqueue(connection):
                        continue
                self._start_serve(connection)
        finally:
            with self.lock:
                if self.acceptor is threading.current_thread():
                    self.acceptor = None

    def _enqueue(self, connection):
        """Park a connection in the bounded session queue.

        Returns True if the caller already owns the gate (slot was free);
        False when the connection was refused (bound hit, daemon stopping)
        or parked for a queue-pump thread to serve later.
        """
        with self.lock:
            if self.stopping.is_set():
                connection.close()
                return False
            self.queue.append(connection)
            if len(self.queue) > self.session_queue:
                # Bound hit: refuse the NEWEST client; earlier arrivals keep
                # their FIFO position.
                evicted = self.queue.pop()
                evicted.close()
                return False
            pumping = self.pumping
            self.pumping = True
        if not pumping:
            threading.Thread(target=self._pump_queue, daemon=True).start()
        return False

    def _pump_queue(self):
        """Hand queued connections to the session slot in FIFO order.

        One pump thread serves the whole queue: it waits for the gate, pops
        the head, and starts a session thread; the session thread releases
        the gate when it ends (its own finally path). The pump exits when
        the queue drains or the daemon stops.
        """
        while True:
            with self.lock:
                if self.stopping.is_set() or not self.queue:
                    self.pumping = False
                    return
                connection = self.queue[0]
            if not self.gate.acquire(timeout=self.timeout):
                # Waiter timeout: drop the head client, keep the queue.
                with self.lock:
                    if self.queue:
                        self.queue.popleft().close()
                continue
            with self.lock:
                if self.stopping.is_set():
                    connection.close()
                    self.gate.release()
                    self.pumping = False
                    return
                if self.queue and self.queue[0] is connection:
                    self.queue.popleft()
                else:
                    # Head changed underneath (dropped by a timeout race);
                    # give the slot back and re-check.
                    self.gate.release()
                    continue
            self._start_serve(connection)

    def _start_serve(self, connection):
        """Register and start one session thread; no leaked slot on failure."""
        thread = threading.Thread(target=self._serve, args=(connection,),
                                  daemon=True)
        with self.lock:
            if self.stopping.is_set():
                connection.close()
                self.gate.release()
                return
            self.workers[thread] = connection
            try:
                thread.start()
            except BaseException:
                # Match the quota authority: no leaked slot or socket when
                # the session thread cannot start.
                self.workers.pop(thread, None)
                connection.close()
                self.gate.release()
                raise

    def _session_stage(self):
        """A fresh worker stage per session.

        The stage protocol closes a stage on STOP, so no stage can outlive
        its session; every session builds its own stage through the factory
        and the construction-time validation copy stays closed. A factory
        returning the same closed object twice fails the freshness checks
        here and in HostHybridWorker's constructor, poisoning only the
        session that received it.
        """
        stage = self.stage_factory()
        if stage.closed or getattr(stage, 'failed', False) or not stage.last or stage.first:
            raise ValueError('worker session requires a fresh final-range stage')
        return stage

    def _serve(self, connection):
        """Own one client session on its own thread; failures stay local."""
        wrapped = None
        thread = threading.current_thread()
        try:
            wrapped = wrap_peer(connection, self.context_tls, server_side=True,
                                peer_pin=(self.client_pin,), timeout=self.timeout,
                                cancelled=self.stopping.is_set, alpn=ALPN)
            transport = HostTensorChannel(wrapped, self.identity, 1,
                                          max_frame_bytes=self.max_frame_bytes,
                                          timeout=self.timeout)
            channel = HostStageChannel(transport,
                                       StageState(1, self.context, self.slots,
                                                  context_limit=2 ** 20,
                                                  models=self.table_size),
                                       vocab_size=self.vocab_size)
            worker = HostHybridWorker(
                channel, self._session_stage(),
                additional_models=[(alias, model) for alias, model
                                   in (self.models or {}).items()
                                   if alias != 'model'])
            worker.handshake()
            # The worker loop: receive/execute/acknowledge until STOP,
            # disconnect, idle timeout or failure. Any error poisons this
            # session only; the daemon returns to idle.
            while not worker.closed and not self.stopping.is_set():
                try:
                    worker.step()
                except (EOFError, OSError, RuntimeError, ValueError):
                    break
            if not worker.closed:
                worker.close()
        except (OSError, EOFError, ValueError, RuntimeError):
            # Handshake/policy/protocol failures refuse or end THIS session;
            # the daemon stays bound and returns to idle. wrap_peer and the
            # channel already closed their side on failure; nothing to do.
            pass
        finally:
            # Closing twice is harmless; the channel may already have closed
            # the transport on its own failure path. Releasing the gate
            # returns the daemon to idle even when cleanup raised.
            with self.lock:
                self.workers.pop(thread, None)
            try:
                self.gate.release()
            except ValueError:
                pass
            if wrapped is not None:
                try:
                    wrapped.close()
                except OSError:
                    pass

    def close(self):
        self.stopping.set()
        try:
            self.listener.close()
        except OSError:
            pass
        with self.lock:
            # Parked queue clients never execute; close them so the pump
            # drains and exits. The pump's final gate handoff is handled by
            # its own stopping check.
            while self.queue:
                try:
                    self.queue.popleft().close()
                except OSError:
                    pass
            threads = list(self.workers)
            for thread in threads:
                # Interrupt any blocked transport from outside; the session
                # thread's own error path finishes the cleanup.
                connection = self.workers.get(thread)
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
            acceptor = self.acceptor
        deadline_threads = [thread for thread in threads
                            if thread is not threading.current_thread()]
        if acceptor is not None and acceptor is not threading.current_thread():
            deadline_threads.append(acceptor)
        deadline = time.monotonic() + self.close_timeout
        for thread in deadline_threads:
            thread.join(max(0., deadline - time.monotonic()))
        with self.lock:
            if self.workers:
                raise RuntimeError('worker daemon session cleanup remains active')
            if (self.acceptor is not None and self.acceptor is not threading.current_thread()
                    and self.acceptor.is_alive()):
                raise RuntimeError('worker daemon acceptor cleanup remains active')
        self.closed = True
