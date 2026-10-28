import json
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from damendaemon import fleet_preflight
from damendaemon.distributed_agreement import (compare_payloads, session_payload,
                                               transport_identity)
from damendaemon.distributed_supervised_builder import (remote_worker_pids,
                                                        supervised_builder,
                                                        teardown_supervised_pair)
from damendaemon.distributed_supervisor import (DistributedSupervisor,
                                                SupervisorRuntime, classify_failure)
from damendaemon.stage_transport import (PeerDeadlineExceeded,
                                          abort_dead_group_bounded,
                                          abort_process_group_safely,
                                          bounded_exchange,
                                          poll_works)


class FakeRuntime:
    def __init__(self, behavior):
        self.behavior = behavior
        self.closed_called = False
        self.run_count = 0

    def run(self, idle_timeout=0):
        self.run_count += 1
        outcome = self.behavior.pop(0) if self.behavior else 'clean'
        if outcome == 'clean-block':
            threading.Event().wait(2)
            return 'clean'
        if outcome == 'timeout':
            raise TimeoutError('transport frame deadline exceeded')
        if outcome == 'nccl':
            raise RuntimeError('NCCL error: remote peer aborted')
        if outcome == 'peer-dead-heartbeat':
            raise PeerDeadlineExceeded(
                'peer heartbeat timeout: polled NCCL exchange did not '
                'complete within 10s deadline')
        if outcome == 'other':
            raise ValueError('logic bug')
        return 'clean'

    def close(self):self.closed_called = True
    def manage(self, action, **options):return dict(action=action, **options)
    def model_ids(self):return ['a']
    def scheduler_status(self):return dict(closed=False)


class FakePair:
    def __init__(self):self.stopped = False;self.closed = False
    def stop(self):self.stopped = True
    def close(self):self.closed = True


class AgreementTest(unittest.TestCase):
    def test_transport_identity_reports_build_and_backend(self):
        fake = mock.Mock()
        fake.__version__ = '2.6.0+cu124'
        fake.version.cuda = '12.4'
        fake.version.hip = None
        fake.cuda.nccl.version.return_value = (2, 21, 5)
        identity = transport_identity(fake)
        self.assertEqual(identity, dict(torch='2.6.0+cu124', backend='nccl',
                                        runtime='12.4', transport='2.21.5'))

    def test_compare_payloads_passes_on_identical(self):
        payload = session_payload([('a', 'ff')], dict(context=512), dict(torch='x'), 4)
        self.assertEqual(compare_payloads(payload, payload)['bounds'], dict(context=512))

    def test_transport_mismatch_names_both_sides(self):
        local = session_payload([], {}, dict(torch='2.6.0', transport='2.21.5'), 4)
        peer = session_payload([], {}, dict(torch='2.13.0', transport='2.29.7'), 4)
        with self.assertRaises(ValueError) as caught:
            compare_payloads(local, peer, local_rank=0)
        message = str(caught.exception)
        self.assertIn('transport', message)
        self.assertIn("2.21.5", message)
        self.assertIn("2.29.7", message)
        self.assertIn('align venvs', message)

    def test_bounds_mismatch_is_actionable(self):
        local = session_payload([('a', 'ff')], dict(context=512), dict(torch='x'), 4)
        peer = session_payload([('a', 'ff')], dict(context=256), dict(torch='x'), 4)
        with self.assertRaises(ValueError) as caught:
            compare_payloads(local, peer)
        self.assertIn('bounds', str(caught.exception))


class PreflightTest(unittest.TestCase):
    def make_torch(self, *, hip=None):
        version = type('version', (), {})()
        version.cuda = None if hip else '12.4'
        version.hip = hip
        cuda = type('cuda', (), {})()
        cuda.is_available = staticmethod(lambda: False)
        cuda.device_count = staticmethod(lambda: 0)
        cuda.nccl = type('nccl', (), {})()
        cuda.nccl.version = staticmethod(lambda: (2, 21, 5))
        return type('torch', (), dict(__version__='2.6.0+cu124', version=version, cuda=cuda))()

    def test_run_report_collects_items_and_overall_pass(self):
        fake = self.make_torch()
        report = fleet_preflight.run_report(fake, peer=None)
        self.assertEqual([item['item'] for item in report['items']], ['torch', 'gpus'])
        torch_item = report['items'][0]
        self.assertEqual(torch_item['build'], 'cuda')
        self.assertEqual(torch_item['nccl'], '2.21.5')

    def test_rocm_build_reports_rccl_note(self):
        fake = self.make_torch(hip='6.4')
        report = fleet_preflight.run_report(fake, peer=None)
        self.assertEqual(report['items'][0]['build'], 'rocm')
        self.assertIn('rccl', report['items'][0]['nccl'])

    def test_peer_probe_classifies_timeout(self):
        fake = self.make_torch(hip='6.4')
        with mock.patch.object(fleet_preflight.socket, 'create_connection',
                               side_effect=__import__('socket').timeout):
            report = fleet_preflight.run_report(fake, peer='10.40.100.30:29500')
        peer = report['items'][-1]
        self.assertFalse(peer['passed'])
        self.assertIn('open-loop hang', peer['note'])
        self.assertFalse(report['passed'])


class FakeWorker:
    def __init__(self):self.killed = False
    def terminate(self):self.killed = True


class FakeWork:
    """CPU-bus stand-in for an async collective handle."""

    def __init__(self, completed_after=0):
        self.completed_after = completed_after
        self.polls = 0

    def is_completed(self):
        self.polls += 1
        return self.polls > self.completed_after


class PolledHeartbeatTest(unittest.TestCase):
    def test_poll_works_returns_when_all_complete(self):
        clock = {'now': 0.0}
        sleeps = []

        def fake_sleep(_):
            sleeps.append(clock['now'])
            clock['now'] += 0.05

        works = [FakeWork(completed_after=2), FakeWork(completed_after=0)]
        poll_works(works, deadline=5, poll_interval=0.05,
                   clock=lambda: clock['now'], sleep=fake_sleep)
        self.assertGreater(works[0].polls, 2)

    def test_poll_works_raises_peer_deadline_on_miss(self):
        clock = {'now': 0.0}

        def fake_sleep(_):
            clock['now'] += 0.05

        work = FakeWork(completed_after=10 ** 9)
        with self.assertRaises(PeerDeadlineExceeded) as raised:
            poll_works([work], deadline=5, poll_interval=0.05,
                       clock=lambda: clock['now'], sleep=fake_sleep)
        message = str(raised.exception)
        self.assertIn('timeout', message)
        # The classification chain must map it to a dead peer.
        self.assertEqual(classify_failure(raised.exception), 'peer-dead')

    def test_poll_works_bounds(self):
        with self.assertRaises(ValueError):
            poll_works([FakeWork()], deadline=0)
        with self.assertRaises(ValueError):
            poll_works([], deadline=1)

    def test_peer_deadline_exception_is_transport_marked(self):
        # The message deliberately carries a transport-timeout marker so the
        # supervisor classification stays pure string/chain based.
        error = PeerDeadlineExceeded('peer heartbeat timeout: polled')
        self.assertEqual(classify_failure(error), 'peer-dead')

    def test_abort_process_group_safely_best_effort(self):
        # Live-measured (2026-10-28): without the abort, the supervisor's
        # destroy_process_group teardown froze the whole coordinator (GIL
        # held) on an errored group. The helper must never raise itself.
        with mock.patch('damendaemon.stage_transport._abort_process_group',
                        return_value=None) as abort:
            self.assertTrue(abort_process_group_safely())
            abort.assert_called_once_with()
        with mock.patch('damendaemon.stage_transport._abort_process_group',
                        side_effect=RuntimeError('no group')):
            self.assertFalse(abort_process_group_safely())
        with mock.patch('damendaemon.stage_transport._abort_process_group',
                        None):
            self.assertFalse(abort_process_group_safely())

    def test_bounded_exchange_returns_completed_result(self):
        clock = {'now': 0.0}

        def fake_sleep(_):
            clock['now'] += 0.05

        self.assertEqual(
            bounded_exchange(lambda: 'ok', 5, poll_interval=0.05,
                             clock=lambda: clock['now'], sleep=fake_sleep),
            'ok')

    def test_bounded_exchange_deadline_on_stuck_post(self):
        # The live finding: the async collective POST itself can block when
        # the peer just died — the bound must classify even then.
        import threading as threading_module
        stuck = threading_module.Event()
        clock = {'now': 0.0}

        def fake_sleep(_):
            clock['now'] += 0.05

        def blocked_post():
            stuck.wait(10)  # simulate the NCCL lock; released after test

        try:
            with self.assertRaises(PeerDeadlineExceeded) as raised:
                bounded_exchange(blocked_post, 1, poll_interval=0.05,
                                 clock=lambda: clock['now'], sleep=fake_sleep)
            self.assertIn('timeout', str(raised.exception))
            self.assertEqual(classify_failure(raised.exception), 'peer-dead')
        finally:
            stuck.set()

    def test_bounded_exchange_propagates_exchange_error(self):
        def broken():
            raise RuntimeError('collective failed')

        with self.assertRaises(RuntimeError):
            bounded_exchange(broken, 5)

    def test_bounded_exchange_bounds(self):
        with self.assertRaises(ValueError):
            bounded_exchange(lambda: None, 0)
        with self.assertRaises(ValueError):
            bounded_exchange('not-callable', 5)

    def test_abort_dead_group_bounded_leaks_hung_abort(self):
        # Gen-2 live finding: the abort itself can block; the caller must
        # not wait on it (the supervisor stayed 'serving' until the group
        # watchdog killed the process). A hung abort returns False fast.
        import threading as threading_module
        release = threading_module.Event()
        clock = {'started': time.monotonic()}

        def hung_abort():
            release.wait(30)

        with mock.patch('damendaemon.stage_transport.'
                        'abort_process_group_safely', side_effect=hung_abort):
            try:
                started = time.monotonic()
                self.assertFalse(abort_dead_group_bounded(bound=0.3))
                self.assertLess(time.monotonic() - started, 5)
            finally:
                release.set()
        with mock.patch('damendaemon.stage_transport.'
                        'abort_process_group_safely', return_value=True):
            self.assertTrue(abort_dead_group_bounded(bound=5))


class RemoteTeardownTest(unittest.TestCase):
    def test_remote_worker_pids_environ_verifies_each_candidate(self):
        # The remote kill must be PID-exact: candidates from the module name
        # in the command line, then /proc environ must carry this
        # generation's run id verbatim — a stale PID cannot pass.
        def fake_run(command, capture_output, text, timeout, check):
            self.assertEqual(command[:2], ['ssh', 'worker-box'])
            script = command[2]
            self.assertIn('TORCHELASTIC_RUN_ID=supervised-123-2', script)
            self.assertIn('/proc/$pid/environ', script)
            self.assertIn('pgrep -f damendaemon.distributed_worker', script)
            return SimpleNamespace(stdout='4242\nbad\n', returncode=0)

        with mock.patch('damendaemon.distributed_supervised_builder.'
                        'subprocess.run', side_effect=fake_run):
            pids = remote_worker_pids('worker-box', 'supervised-123-2')
        self.assertEqual(pids, [4242])

    def test_remote_worker_pids_swallows_transport_failure(self):
        # A dead SSH path must not raise into the supervisor teardown.
        def fake_run(*args, **kwargs):
            raise OSError('ssh unreachable')

        with mock.patch('damendaemon.distributed_supervised_builder.'
                        'subprocess.run', side_effect=fake_run):
            self.assertEqual(
                remote_worker_pids('worker-box', 'supervised-123-2'), [])

    def test_remote_teardown_kills_verified_pids_only(self):
        kills = []

        def fake_run(command, capture_output, text=False, timeout=None,
                     check=False):
            if 'pgrep' in command[2]:  # environ-verified probe
                return SimpleNamespace(stdout='777\n', returncode=0)
            kills.append(command)
            return SimpleNamespace(stdout='', returncode=0)

        pair = SimpleNamespace(supervised_remote=dict(host='worker-box',
                                                      run_id='supervised-9-1'),
                               supervised_worker=None,
                               supervised_store=None)
        with mock.patch('damendaemon.distributed_supervised_builder.'
                        'subprocess.run', side_effect=fake_run):
            teardown_supervised_pair(pair)
        self.assertEqual(kills, [['ssh', 'worker-box', 'kill 777']])

    def test_cross_host_builder_requires_every_knob(self):
        args = SimpleNamespace(worker_host='worker-box', worker_gpu=None,
                               worker_python=None, worker_repo=None,
                               worker_model_path=None,
                               rendezvous_host=None,
                               coordinator_ifname=None, worker_ifname=None,
                               model=[], heartbeat_deadline=10.0,
                               path='/m', context=512, quantization='none',
                               alias='model', batch_size=1, queue_size=8,
                               queue_timeout=120, client_decode_turns=False)
        with self.assertRaises(ValueError) as raised:
            supervised_builder(args, torch)
        self.assertIn('cross-host', str(raised.exception))
        # Additional resident models cannot be mirrored on the remote.
        args.model = ['extra=/m2']
        args.worker_gpu = 'GPU-abc'
        args.worker_python = '/p'
        args.worker_repo = '/r'
        args.rendezvous_host = '10.30.100.3'
        args.coordinator_ifname = 'enp6s18'
        args.worker_ifname = 'enp72s0'
        with self.assertRaises(ValueError) as raised:
            supervised_builder(args, torch)
        self.assertIn('loopback-only', str(raised.exception))


class SupervisorTest(unittest.TestCase):
    def make(self, behaviors, builds=None):
        pair, runtime = FakePair(), FakeRuntime(list(behaviors))
        self.builders = builds if builds is not None else 8
        def builder(generation):
            if generation > self.builders:
                raise RuntimeError('no more workers')
            return pair, runtime
        return DistributedSupervisor(builder, max_generations=3), pair, runtime

    def wait_state(self, supervisor, states, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if supervisor.state in states:return True
            time.sleep(0.01)
        return False

    def test_start_serving_and_clean_close(self):
        supervisor, pair, runtime = self.make(['clean-block'])
        supervisor.start()
        self.assertEqual(supervisor.state, 'serving')
        self.assertEqual(supervisor.generation, 1)
        supervisor.close()
        self.assertEqual(supervisor.state, 'closed')
        self.assertTrue(runtime.closed_called and pair.stopped and pair.closed)

    def test_dead_peer_detected_and_rejoin_resumes(self):
        supervisor, pair, runtime = self.make(['timeout', 'clean-block'])
        supervisor.start()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        self.assertIn('TimeoutError', supervisor.reason)
        health = supervisor.runtime_proxy.health()
        self.assertEqual(health[0], 503)
        self.assertTrue(supervisor.rejoinable())
        initiated, detail = supervisor.runtime_proxy.rejoin()
        self.assertTrue(initiated)
        self.assertTrue(self.wait_state(supervisor, {'serving'}))
        self.assertEqual(supervisor.generation, 2)
        self.assertIn('not replayed', detail)
        supervisor.close()
        self.assertEqual(supervisor.state, 'closed')

    def test_rejoin_bound_reaches_terminal(self):
        supervisor, pair, runtime = self.make(['timeout', 'timeout', 'timeout'], builds=3)
        supervisor.start()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        supervisor.runtime_proxy.rejoin()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        initiated, _ = supervisor.runtime_proxy.rejoin()
        self.assertTrue(initiated)
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        initiated, detail = supervisor.runtime_proxy.rejoin()
        self.assertFalse(initiated)
        self.assertIn('exhausted', detail)
        self.assertEqual(supervisor.state, 'terminal')

    def test_build_failure_after_exhaustion_is_terminal(self):
        supervisor, pair, runtime = self.make(['timeout'], builds=1)
        supervisor.start()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        initiated, _ = supervisor.runtime_proxy.rejoin()
        self.assertTrue(initiated)
        self.assertTrue(self.wait_state(supervisor, {'failed'}))
        initiated, _ = supervisor.runtime_proxy.rejoin()
        self.assertTrue(initiated)
        self.assertTrue(self.wait_state(supervisor, {'terminal'}))

    def test_nccl_failure_classified_peer_dead(self):
        supervisor, pair, runtime = self.make(['nccl', 'clean-block'])
        supervisor.start()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        self.assertEqual(classify_failure(ValueError('logic bug')), 'failed')

    def test_polled_heartbeat_death_engages_rejoin(self):
        # The #637 finding closed: a peer death on the idle heartbeat now
        # surfaces as a catchable PeerDeadlineExceeded (the polled exchange
        # never blocks on the dead group), so the supervisor classifies,
        # keeps HTTP honest, and the operator rejoin reaches a fresh
        # generation instead of the NCCL watchdog killing the process.
        supervisor, pair, runtime = self.make(['peer-dead-heartbeat',
                                               'clean-block'])
        supervisor.start()
        self.assertTrue(self.wait_state(supervisor, {'peer-dead'}))
        self.assertIn('PeerDeadlineExceeded', supervisor.reason)
        health = supervisor.runtime_proxy.health()
        self.assertEqual(health[0], 503)
        self.assertEqual(health[1]['status'], 'peer-dead')
        self.assertTrue(supervisor.rejoinable())
        initiated, detail = supervisor.runtime_proxy.rejoin()
        self.assertTrue(initiated)
        self.assertTrue(self.wait_state(supervisor, {'serving'}))
        self.assertEqual(supervisor.generation, 2)
        supervisor.close()

    def test_rejoin_rejected_while_serving(self):
        supervisor, pair, runtime = self.make(['clean-block'])
        runtime.run = lambda idle_timeout=0: threading.Event().wait(10)
        supervisor.start()
        initiated, detail = supervisor.runtime_proxy.rejoin()
        self.assertFalse(initiated)
        self.assertIn('serving', detail)
        supervisor.close()

    def test_manage_status_reports_supervisor_state(self):
        supervisor, pair, runtime = self.make(['clean-block'])
        runtime.run = lambda idle_timeout=0: threading.Event().wait(10)
        supervisor.start()
        status = supervisor.runtime_proxy.manage('status')
        self.assertEqual(status['supervisor']['state'], 'serving')
        supervisor.close()


if __name__ == '__main__':
    unittest.main()
