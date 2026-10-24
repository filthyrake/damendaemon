import json
import threading
import time
import unittest
from unittest import mock

import torch

from damendaemon import fleet_preflight
from damendaemon.distributed_agreement import (compare_payloads, session_payload,
                                               transport_identity)
from damendaemon.distributed_supervisor import (DistributedSupervisor,
                                                SupervisorRuntime, classify_failure)


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
