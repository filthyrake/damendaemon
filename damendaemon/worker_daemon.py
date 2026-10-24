"""Bounded listener daemon exposing this host's stage to a remote peer (#603).

No silent enrollment: nothing binds until a WorkerDaemon is explicitly
constructed, and construction fails unless the configured model loads and
fingerprints cleanly. The daemon speaks the EXISTING qualified two-rank
protocol — HostTensorChannel frames + HostStageChannel commands through a
HostHybridWorker — over the existing mutual-TLS machinery with certificate
pins and the damendaemon-host/1 ALPN; no second protocol is invented.

Session lifecycle: one exclusive client session at a time. A client that
arrives while the session slot is held is refused at accept time; it never
queues on the socket. Each session gets a fresh stage through the factory
and any protocol, execution or handshake failure poisons that session
only, returning the daemon to idle. Client disconnect and idle timeouts
end the session through the channel's own transport discipline. No
multi-tenant scheduling, no automatic restart, no serving admission and
no throughput claims; the caller supplies the bind address, and loopback
is the safe default for testing.
"""
import socket
import threading
import time

from .host_hybrid_worker import HostHybridWorker
from .host_stage_channel import HostStageChannel
from .host_tensor_channel import HostTensorChannel
from .host_tls import ALPN, certificate_pin, server_context, wrap_peer
from .stage_protocol import StageState


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
                 timeout=30, close_timeout=10):
        if (not isinstance(address, tuple) or len(address) != 2
                or not isinstance(address[0], str) or type(address[1]) is not int
                or not 0 <= address[1] <= 65535
                or not callable(stage_factory)
                or not isinstance(identity, bytes) or len(identity) != 32
                or type(max_frame_bytes) is not int or not 1 <= max_frame_bytes <= 2 ** 31
                or type(timeout) is not int or not 0 < timeout <= 30
                or type(close_timeout) is not int or not 0 < close_timeout <= 30):
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
        # TLS: the existing server context + the single admitted client pin.
        self.context_tls = server_context(ca_file=str(tls_dir / 'ca.pem'),
                                          cert_file=str(tls_dir / 'server.pem'),
                                          key_file=str(tls_dir / 'server.key'),
                                          alpn=ALPN)
        self.client_pin = certificate_pin(client_pin)
        self.timeout = timeout
        self.close_timeout = close_timeout
        # One exclusive session: the gate is a single-slot semaphore, and a
        # busy daemon refuses a new client AT ACCEPT TIME (it never queues).
        self.stopping = threading.Event()
        self.gate = threading.BoundedSemaphore(1)
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
                    # Exclusive session policy: refuse the client now.
                    connection.close()
                    continue
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
                        # Match the quota authority: no leaked slot or socket
                        # when the session thread cannot start.
                        self.workers.pop(thread)
                        connection.close()
                        self.gate.release()
                        raise
        finally:
            with self.lock:
                if self.acceptor is threading.current_thread():
                    self.acceptor = None

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
                                                  context_limit=2 ** 20),
                                       vocab_size=self.vocab_size)
            worker = HostHybridWorker(channel, self._session_stage())
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
