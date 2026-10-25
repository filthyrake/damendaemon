"""Worker daemon tests: the #603 listener over real mutual-TLS loopback.

The daemon wraps the existing qualified two-rank worker protocol in a
bounded listener; these tests hold it to the daemon-specific promises —
construction-time model validation, exclusive session admission (refuse
at accept, or park in the bounded queue), per-session stage freshness,
refusal of untrusted clients, session poisoning without daemon death,
live model switching on a pre-agreed two-model table, and clean close —
against real TLS sockets with ephemeral credentials. Remote-hardware
qualification (a coordinator on another host driving this daemon's GPU
stage) is the #603 cell recorded as evidence, not this file.
"""
import socket
import ssl
import threading
import time
import unittest
from unittest.mock import patch

import torch

from damendaemon.host_hybrid_worker import HostHybridWorker
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.host_tls import (client_context, wrap_peer, certificate_pin)
from damendaemon.hybrid import HybridExecutor
from damendaemon.hybrid_stage import HybridLayerStage
from damendaemon.stage_identity import fingerprint
from damendaemon.stage_protocol import StageState
from damendaemon.worker_daemon import WorkerDaemon
from tests.tls_credentials import credentials
from tests import test_hybrid_stage

FIXTURES = credentials()
PROMPTS = ([1, 2, 3], [4, 5])


class WorkerDaemonTests(unittest.TestCase):

    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.built = []
        self.failures = []
        self.identity = fingerprint(self.fixture.model)
        self.daemon = self.start_daemon()
        self.client_tls = client_context(ca_file=FIXTURES / 'ca.pem',
                                         cert_file=FIXTURES / 'client.pem',
                                         key_file=FIXTURES / 'client.key')
        self.server_pin = certificate_pin(FIXTURES / 'server.pem')

    # -- assembly ----------------------------------------------------------

    def stage_factory(self):
        if self.failures:
            raise self.failures.pop(0)
        stage = self.fixture.stage(2, 4)
        self.built.append(stage)
        return stage

    def start_daemon(self, **overrides):
        options = dict(identity=self.identity, tls_dir=FIXTURES,
                       client_pin=FIXTURES / 'client.pem',
                       max_frame_bytes=1024 ** 2, timeout=5, close_timeout=5)
        options.update(overrides)
        daemon = WorkerDaemon(('127.0.0.1', 0), self.stage_factory, **options)
        self.addCleanup(daemon.close)
        thread = threading.Thread(target=daemon.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 6)
        return daemon

    def connect(self, *, cert='client', stage=None, raw=None, models=(),
                model_table=1, daemon=None):
        """One coordinator (rank 0) over real TLS to the live daemon."""
        address = (self.daemon if daemon is None else daemon).address
        raw = (socket.create_connection(address, timeout=5)
               if raw is None else raw)
        context = (self.client_tls if cert == 'client' else
                   client_context(ca_file=FIXTURES / 'ca.pem',
                                  cert_file=FIXTURES / (cert + '.pem'),
                                  key_file=FIXTURES / (cert + '.key')))
        wrapped = wrap_peer(raw, context, server_side=False,
                            peer_pin=self.server_pin,
                            server_hostname='localhost', timeout=5)
        self.addCleanup(wrapped.close)
        stage = self.fixture.stage(0, 2) if stage is None else stage
        self.addCleanup(stage.close)
        transport = HostTensorChannel(wrapped, self.identity, 0,
                                      max_frame_bytes=1024 ** 2, timeout=5)
        channel = HostStageChannel(transport,
                                   StageState(1, stage.context, stage.max_batch,
                                              models=model_table),
                                   vocab_size=stage.c['vocab_size'])
        return HostHybridWorker(channel, stage, additional_models=models)

    def session(self, *, cert='client', daemon=None):
        coordinator = self.connect(cert=cert, daemon=daemon)
        coordinator.handshake()
        self.addCleanup(coordinator.close)
        return coordinator

    def wait_idle(self, daemon=None):
        deadline = time.monotonic() + 5
        target = self.daemon if daemon is None else daemon
        while target.workers and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(target.workers)

    def alt_model(self):
        """A second model with the same config but different weights.

        The fixture's model() re-uses its prepared tensors (`.float()` is
        identity on FP32), so the clone must be explicit or the perturbation
        would reach the primary model too.
        """
        source = self.fixture.fixture.fixture
        model = source.model()
        model.weights = {name: tensor.clone()
                         for name, tensor in model.weights.items()}
        model.weights['head'] = model.weights['embed']  # keep the loader's tie
        rng = torch.Generator().manual_seed(7)
        with torch.no_grad():
            weight = model.weights['embed']  # tied head: shifts the output logits decisively
            weight.add_(torch.randn(weight.shape, generator=rng))
        return model

    def alt_stage_factory(self, model):
        def factory():
            with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                    patch('damendaemon.hybrid.empty_cache'), \
                    patch('damendaemon.hybrid.available_memory', return_value=2 ** 40):
                return HybridLayerStage(model, 2, 4, max_batch=2)
        return factory

    def alt_executor(self, model):
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2 ** 40):
            return HybridExecutor(model, use_graph=False, max_batch=2,
                                  kernels='torch')

    # -- construction ------------------------------------------------------

    def test_construction_loads_validates_and_releases_without_serving(self):
        self.assertEqual(len(self.built), 1)
        self.assertTrue(self.built[0].closed)
        self.assertEqual(len(self.daemon.identity), 32)
        self.assertEqual(self.daemon.address[0], '127.0.0.1')
        self.assertNotEqual(self.daemon.address[1], 0)

    def test_invalid_configuration_is_refused_before_anything_loads(self):
        calls = []

        def factory():
            calls.append(1)
            return self.fixture.stage(2, 4)
        base = dict(tls_dir=FIXTURES, client_pin=FIXTURES / 'client.pem',
                    identity=fingerprint(self.fixture.model))
        cases = [(('x',), {}), (('127.0.0.1', True), {}),
                 (('127.0.0.1', 0), {'identity': b'short'}),
                 (('127.0.0.1', 0), {'identity': None}),
                 (('127.0.0.1', 0), {'identity': 'x' * 32}),
                 (('127.0.0.1', 0), {'max_frame_bytes': 0}),
                 (('127.0.0.1', 0), {'max_frame_bytes': 2 ** 31 + 1}),
                 (('127.0.0.1', 0), {'max_frame_bytes': None}),
                 (('127.0.0.1', 0), {'timeout': 0}),
                 (('127.0.0.1', 0), {'timeout': 31}),
                 (('127.0.0.1', 0), {'timeout': 1.5}),
                 (('127.0.0.1', 0), {'close_timeout': 0}),
                 (('127.0.0.1', 0), {'close_timeout': 31})]
        for address, override in cases:
            with self.assertRaises(ValueError):
                WorkerDaemon(address, factory, **(base | override))
        self.assertFalse(calls)

    def test_construction_rejects_first_range_and_broken_models(self):
        # A factory that yields a FIRST-range stage is not a daemon head.
        with self.assertRaisesRegex(ValueError, 'final-range'):
            WorkerDaemon(('127.0.0.1', 0), lambda: self.fixture.stage(0, 2),
                         identity=fingerprint(self.fixture.model),
                         tls_dir=FIXTURES, client_pin=FIXTURES / 'client.pem')
        # A broken factory fails construction; nothing binds afterwards.
        def broken():
            raise RuntimeError('checkpoint unreadable')
        with self.assertRaisesRegex(RuntimeError, 'unreadable'):
            WorkerDaemon(('127.0.0.1', 0), broken, tls_dir=FIXTURES,
                         client_pin=FIXTURES / 'client.pem',
                         identity=fingerprint(self.fixture.model))

    # -- sessions ----------------------------------------------------------

    def test_session_matches_single_executor_and_daemon_survives_reuse(self):
        full = self.fixture.fixture.fixture.executor()
        for prompt in PROMPTS:
            expected = full.prefill(list(prompt)).argmax(-1).tolist()[0]
            coordinator = self.session()
            self.assertEqual(coordinator.prefill(list(prompt)), expected)
            coordinator.stop()
            self.wait_idle()
        # Validation copy + one per session; every session stage closed.
        self.assertEqual(len(self.built), 1 + len(PROMPTS))
        self.assertTrue(all(stage.closed for stage in self.built))

    def test_busy_session_refuses_second_client_at_accept(self):
        first = self.session()
        second = socket.create_connection(self.daemon.address, timeout=5)
        self.addCleanup(second.close)
        with self.assertRaises((ValueError, OSError, EOFError, ssl.SSLError)):
            wrap_peer(second, self.client_tls, server_side=False,
                      peer_pin=self.server_pin, server_hostname='localhost',
                      timeout=5)
        # The busy refusal did not disturb the live session.
        expected = self.fixture.fixture.fixture.executor().prefill([7]).argmax(-1).tolist()[0]
        self.assertEqual(first.prefill([7]), expected)
        first.stop()
        self.wait_idle()
        again = self.session()
        self.assertEqual(again.prefill([7]), expected)
        again.stop()
        self.wait_idle()

    def test_untrusted_client_is_refused_and_daemon_survives(self):
        with self.assertRaises((ValueError, OSError, EOFError, ssl.SSLError)):
            self.session(cert='untrusted')
        self.wait_idle()
        good = self.session()
        expected = self.fixture.fixture.fixture.executor().prefill([8]).argmax(-1).tolist()[0]
        self.assertEqual(good.prefill([8]), expected)
        good.stop()
        self.wait_idle()

    def test_session_failure_poisons_only_that_session(self):
        expected = self.fixture.fixture.fixture.executor().prefill([9]).argmax(-1).tolist()[0]
        good = self.session()
        self.assertEqual(good.prefill([9]), expected)
        good.stop()
        self.wait_idle()
        # Next session's factory failure refuses the client, not the daemon.
        self.failures.append(RuntimeError('session model failed to load'))
        refused = self.connect()
        with self.assertRaises((ValueError, OSError, EOFError, RuntimeError)):
            refused.handshake()
        self.wait_idle()
        # The daemon still serves a well-formed session afterwards.
        good = self.session()
        self.assertEqual(good.prefill([9]), expected)
        good.stop()
        self.wait_idle()

    # -- bounded session queue ----------------------------------------------

    def test_bounded_queue_serves_parked_client_and_refuses_newest(self):
        daemon = self.start_daemon(session_queue=1)
        first = self.session(daemon=daemon)
        # One park slot: the second client is accepted and held, not served.
        parked = socket.create_connection(daemon.address, timeout=5)
        self.addCleanup(parked.close)
        deadline = time.monotonic() + 2
        while len(daemon.queue) < 1 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(daemon.queue), 1)
        # The queue bound is hit by a third client: the NEWEST is refused.
        refused = socket.create_connection(daemon.address, timeout=5)
        self.addCleanup(refused.close)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if refused.recv(1) == b'':
                    break
            except OSError:
                break
        else:
            self.fail('overflow client was not refused')
        # The live session is undisturbed; when it ends, the pump serves the
        # parked client in FIFO order and it executes correctly.
        expected = self.fixture.fixture.fixture.executor().prefill([7]).argmax(-1).tolist()[0]
        self.assertEqual(first.prefill([7]), expected)
        first.stop()
        deadline = time.monotonic() + 5
        while daemon.queue and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(daemon.queue)
        second = self.connect(raw=parked, daemon=daemon)
        second.handshake()
        self.assertEqual(second.prefill([7]), expected)
        second.stop()
        self.wait_idle(daemon)

    def test_parked_client_is_dropped_after_the_wait_timeout(self):
        daemon = self.start_daemon(session_queue=1, timeout=1)
        first = self.session(daemon=daemon)
        parked = socket.create_connection(daemon.address, timeout=5)
        self.addCleanup(parked.close)
        deadline = time.monotonic() + 2
        while len(daemon.queue) < 1 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(daemon.queue), 1)
        # The gate never frees within the 1s waiter timeout. Keep the live
        # session transport alive with pings while the wait expires: the
        # daemon timeout is ALSO the session idle timeout.
        deadline = time.monotonic() + 3
        while len(daemon.queue) and time.monotonic() < deadline:
            first.ping()
            time.sleep(.2)
        self.assertEqual(len(daemon.queue), 0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if parked.recv(1) == b'':
                    break
            except OSError:
                break
        else:
            self.fail('timed-out parked client was not closed')
        # The daemon still serves: the live session ends and a fresh one works.
        expected = self.fixture.fixture.fixture.executor().prefill([7]).argmax(-1).tolist()[0]
        self.assertEqual(first.prefill([7]), expected)
        first.stop()
        self.wait_idle(daemon)
        again = self.session(daemon=daemon)
        self.assertEqual(again.prefill([7]), expected)
        again.stop()
        self.wait_idle(daemon)

    # -- live model switching ------------------------------------------------

    def test_model_switch_on_live_session_round_trips_both_models(self):
        alt = self.alt_model()
        daemon = self.start_daemon(additional_models=[('alt', self.alt_stage_factory(alt))])
        expected_base = self.fixture.fixture.fixture.executor().prefill([7]).argmax(-1).tolist()[0]
        expected_alt = self.alt_executor(alt).prefill([7]).argmax(-1).tolist()[0]
        self.assertNotEqual(expected_base, expected_alt)
        coordinator = self.connect(models=[('alt', alt)], model_table=2, daemon=daemon)
        coordinator.handshake()
        self.addCleanup(coordinator.close)
        self.assertTrue(coordinator.switch_model('alt'))
        self.assertFalse(coordinator.switch_model('alt'))  # already current
        self.assertEqual(coordinator.prefill([7]), expected_alt)
        # An occupied slot refuses a switch; release clears it.
        with self.assertRaises(ValueError):
            coordinator.switch_model('model')
        coordinator.release([0])
        self.assertTrue(coordinator.switch_model('model'))
        self.assertEqual(coordinator.prefill([7]), expected_base)
        coordinator.stop()
        self.wait_idle(daemon)
        # A fresh session opens on the primary model again.
        fresh = self.connect(models=[('alt', alt)], model_table=2, daemon=daemon)
        fresh.handshake()
        self.addCleanup(fresh.close)
        self.assertEqual(fresh.prefill([7]), expected_base)
        fresh.stop()
        self.wait_idle(daemon)

    def test_model_table_mismatch_refuses_the_session(self):
        alt = self.alt_model()
        daemon = self.start_daemon(additional_models=[('alt', self.alt_stage_factory(alt))])
        with self.assertRaises((ValueError, OSError, EOFError, RuntimeError)):
            self.connect(daemon=daemon).handshake()  # client table: models=1
        self.wait_idle(daemon)
        # A client with the matching table is admitted; a matching-table
        # session serves the primary model normally.
        good = self.connect(models=[('alt', alt)], model_table=2, daemon=daemon)
        good.handshake()
        self.addCleanup(good.close)
        expected = self.fixture.fixture.fixture.executor().prefill([7]).argmax(-1).tolist()[0]
        self.assertEqual(good.prefill([7]), expected)
        good.stop()
        self.wait_idle(daemon)

    def test_invalid_model_table_is_refused_before_binding(self):
        alt = self.alt_model()
        # A mismatched decoder config cannot enter the switch table.
        other = self.fixture.fixture.fixture.model()
        other.config = {**other.config, 'num_hidden_layers': 2}
        for factory in (lambda: self.fixture.stage(0, 2),  # not final-range
                        lambda: HybridLayerStage(other, 2, 2, max_batch=2)):
            with self.assertRaises(ValueError):
                WorkerDaemon(('127.0.0.1', 0), self.stage_factory,
                             identity=self.identity, tls_dir=FIXTURES,
                             client_pin=FIXTURES / 'client.pem',
                             additional_models=[('alt', factory)])

    # -- shutdown ------------------------------------------------------------

    def test_close_interrupts_active_session_and_listener(self):
        coordinator = self.session()
        self.assertTrue(self.daemon.workers)
        started = time.monotonic()
        self.daemon.close()
        self.assertLess(time.monotonic() - started, 6)
        self.assertTrue(self.daemon.closed)
        self.assertFalse(self.daemon.workers)
        with self.assertRaises((OSError, EOFError, RuntimeError, ValueError)):
            coordinator.prefill([1])
        with self.assertRaisesRegex(RuntimeError, 'serving or stopping'):
            self.daemon.serve_forever()


if __name__ == '__main__':
    unittest.main()
