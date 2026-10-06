import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments.cuda_capacity_guard import A30CapacityGuard


class CudaCapacityGuardTests(unittest.TestCase):
    def test_preflight_and_live_thermal_cutoff(self):
        report = {}
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-test'), \
                patch('scripts.experiments.cuda_capacity_guard.host_available', return_value=110*1024**3), \
                patch('scripts.experiments.cuda_capacity_guard.time.monotonic', side_effect=[0, .5, 2]), \
                patch('scripts.experiments.cuda_capacity_guard.subprocess.run', side_effect=[
                    SimpleNamespace(stdout='NVIDIA A30, 46, 235, 23930'),
                    SimpleNamespace(stdout='NVIDIA A30, 85, 21000, 3100')]) as query:
            guard = A30CapacityGuard(report)
            self.assertEqual(guard.read(startup=True), 110*1024**3)
            guard.read()
            self.assertEqual(query.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, 'thermal cutoff'):
                guard.read()
        self.assertEqual(report['maximum_observed_celsius'], 85)
        self.assertEqual(report['maximum_device_used_bytes'], 21000*1024**2)
        self.assertIn('--id=GPU-test', query.call_args.args[0])

    def test_missing_or_unsafe_readback_fails_closed(self):
        for reading, available in (('NVIDIA A30, 46, 8000, 16000', 110),
                                   ('NVIDIA A30, 46, 235, 23930', 99),
                                   ('NVIDIA A30, 70, 235, 23930', 110),
                                   ('NVIDIA A2, 46, 235, 23930', 110),
                                   ('NVIDIA A30, N/A, 235, 23930', 110)):
            with self.subTest(reading=reading, available=available), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-test'), \
                    patch('scripts.experiments.cuda_capacity_guard.host_available', return_value=available*1024**3), \
                    patch('scripts.experiments.cuda_capacity_guard.subprocess.run', return_value=SimpleNamespace(stdout=reading)), \
                    self.assertRaises((RuntimeError, MemoryError, ValueError)):
                A30CapacityGuard({}).read(startup=True)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-a,GPU-b'), self.assertRaises(ValueError):
            A30CapacityGuard({})
