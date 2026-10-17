from types import SimpleNamespace
import unittest
from unittest.mock import patch

from damendaemon.device import selected_device, available_memory
from damendaemon.runtime import require_gpu


class DeviceTests(unittest.TestCase):
    def test_mps_does_not_require_cuda_visibility(self):
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'mps'}, clear=True), patch.dict('sys.modules', {'torch': torch}):
            require_gpu()

    def test_unavailable_mps_fails_without_cpu_fallback(self):
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'mps'}), patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(RuntimeError, 'not available'):
                require_gpu()

    def test_cuda_still_requires_exactly_one_uuid(self):
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda', 'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaises(ValueError):
                require_gpu()

    def test_mps_budget_excludes_driver_allocations(self):
        torch = SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=lambda: 1000,
                                                     driver_allocated_memory=lambda: 800))
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'mps'}), patch.dict('sys.modules', {'torch': torch}):
            self.assertEqual(available_memory(), 200)

    def test_unknown_device_rejected(self):
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cpu'}):
            with self.assertRaises(ValueError):
                selected_device()
