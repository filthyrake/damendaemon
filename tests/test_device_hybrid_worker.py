"""Device-boundary worker tests: CPU rejections plus a real MPS device pair.

CI hosts without an accelerator cover the boundary contract's rejection paths.
Where Apple MPS is present, the full two-rank worker protocol runs over a
real device pair through DeviceHybridStage boundaries and matches the
single-device executor control — the loopback contract lifted to device
hardware. CUDA+XPU cross-vendor qualification is the #600 desktop cell and
is recorded as evidence, not in this file.
"""
import unittest
from unittest.mock import patch

import torch

from damendaemon.device_hybrid_stage import DeviceHybridStage
from damendaemon.host_hybrid_worker import HostHybridWorker
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.stage_protocol import StageState
from tests import test_host_tensor_channel, test_hybrid_stage


class DeviceBoundaryRejectionTests(unittest.TestCase):
    """Contract guards that need no accelerator."""

    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_cpu_stage_is_rejected_with_explicit_reason(self):
        stage = self.fixture.stage(0, 2)
        with self.assertRaisesRegex(ValueError, 'non-CPU stage device'):
            DeviceHybridStage(stage, max_transfer_bytes=1024**3)

    def test_transfer_budget_is_bounded_and_explicit(self):
        stage = self.fixture.stage(0, 2)
        with patch.object(type(stage), 'actual_device',
                         property(lambda self: torch.device('mps'))):
            for budget in (0, 2**31 + 1, -1, None, 'big'):
                with self.assertRaises(ValueError):
                    DeviceHybridStage(stage, max_transfer_bytes=budget)
            boundary = DeviceHybridStage(stage, max_transfer_bytes=2**31)
        self.addCleanup(boundary.close)
        self.assertEqual(boundary.boundary_dtype, torch.float32)
        self.assertFalse(boundary.closed)

    def test_non_stage_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hybrid layer stage'):
            DeviceHybridStage(object(), max_transfer_bytes=1024**3)


@unittest.skipUnless(getattr(torch, 'mps', None) is not None
                     and torch.mps.is_available(), 'device pair requires MPS')
class DeviceWorkerPairTests(unittest.TestCase):
    """The two-rank worker protocol over real device boundary stages."""

    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.mps_model()
        self.control = self.executor()

    def mps_model(self):
        from types import SimpleNamespace
        base = self.fixture.model
        weights = {name: (value.half() if value.is_floating_point() else value)
                   for name, value in base.weights.items()}
        weights['head'] = weights['embed']  # Preserve the tied head identity.
        return SimpleNamespace(config=base.config, max_context=base.max_context,
                               weights=weights, scales={}, quantization='none',
                               bytes=sum(t.numel() * t.element_size() for t in
                                         {id(t): t for t in weights.values()}.values()))

    def device_patches(self):
        return (patch('damendaemon.hybrid.selected_device', return_value='mps'),
                patch('damendaemon.hybrid.empty_cache'),
                patch('damendaemon.hybrid.available_memory', return_value=2**40))

    def stage(self, start, end):
        from damendaemon.hybrid_stage import HybridLayerStage
        with self.device_patches()[0], self.device_patches()[1], self.device_patches()[2]:
            return HybridLayerStage(self.model, start, end, max_batch=2)

    def executor(self):
        from damendaemon.hybrid import HybridExecutor
        with self.device_patches()[0], self.device_patches()[1], self.device_patches()[2]:
            return HybridExecutor(self.model, use_graph=False, max_batch=2, kernels='torch')

    def pair(self, *, caps=(64 * 1024, 64 * 1024), timeout=30):
        stages = [self.stage(0, 2), self.stage(2, 4)]
        boundaries = [DeviceHybridStage(stage, max_transfer_bytes=8 * 1024**2)
                      for stage in stages]
        for boundary in boundaries:
            self.addCleanup(boundary.close)
        transports = test_host_tensor_channel.HostTensorChannelTests.pair(
            self, caps=caps, timeout=timeout)
        channels = [HostStageChannel(transport, StageState(1, boundary.context,
                                                           boundary.max_batch),
                                     vocab_size=boundary.c['vocab_size'])
                    for transport, boundary in zip(transports, boundaries)]
        workers = [HostHybridWorker(channel, boundary)
                   for channel, boundary in zip(channels, boundaries)]
        for worker in workers:
            self.addCleanup(worker.close)
        self.assertTrue(all(worker.device_stage for worker in workers))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda worker: worker.handshake(), workers))
        return workers

    def parallel(self, functions):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            return [future.result(timeout=60) for future in
                    [pool.submit(function) for function in functions]]

    def test_device_pair_matches_single_device_control(self):
        left, right = self.pair()
        self.assertEqual(left.peer_descriptor, right.descriptor)
        self.assertEqual(right.peer_descriptor, left.descriptor)
        prompt = [1, 2, 3]
        expected = self.control.prefill(prompt).argmax(-1).tolist()
        self.assertEqual(self.parallel([lambda: left.prefill(prompt),
                                        right.step])[0], expected[0])
        tokens = [expected[0]]
        for _ in range(3):
            _, ids = self.control.decode_batch(tokens[-1:], [0])
            tokens.extend(ids.tolist())
            self.assertEqual(self.parallel(
                [lambda: left.decode([tokens[-2]], [0]), right.step])[0], ids.tolist())
        self.assertEqual(len(tokens), 4)  # one prefill token plus three decode steps
        # Boundary statistics are recorded for the transfer decomposition.
        stats = left.stage.stats
        self.assertGreater(stats['calls'], 0)
        self.assertGreater(stats['bytes_out'], 0)
        # Rank 0's outbound accounting covers every boundary volume it
        # produced, including the activations rank 1 received.
        self.assertGreaterEqual(stats['bytes_out'], right.stage.stats['bytes_in'])

    def test_boundary_budget_rejection_poisons_the_session(self):
        import concurrent.futures
        left, right = self.pair()
        left.stage.max_transfer_bytes = 8
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(left.prefill, [1, 2, 3]), pool.submit(right.step)]
            for future in futures:
                with self.assertRaises((ValueError, RuntimeError, OSError, EOFError)):
                    future.result(timeout=60)
        self.assertTrue(all(worker.failed and worker.closed
                            for worker in (left, right)))

    def test_release_and_stop_close_device_stages_cleanly(self):
        left, right = self.pair()
        self.parallel([lambda: left.prefill([1, 2]), right.step])
        self.parallel([lambda: left.release([0]), right.step])
        for worker in (left, right):
            for tensor in worker.stage.kv.values():
                self.assertEqual(int(tensor[:, 0].count_nonzero()), 0)
        self.parallel([left.ping, right.step])
        self.parallel([left.stop, right.step])
        self.assertTrue(all(worker.closed and worker.stage.closed
                            for worker in (left, right)))
        self.assertTrue(all(not worker.stage.w for worker in (left, right)))


if __name__ == '__main__':
    unittest.main()
