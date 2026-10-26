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

    def test_cuda_admits_a_single_mig_slice_uuid(self):
        """#609 gate 2: a MIG slice is admitted as a device (single-slice)."""
        torch = SimpleNamespace(version=SimpleNamespace(cuda='13.0', hip=None),
                                cuda=SimpleNamespace(is_available=lambda: True,
                                                     device_count=lambda: 1,
                                                     set_device=lambda index: None))
        visible = 'MIG-c083756b-37b9-51b5-ac0e-6cdda07b784c'
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda',
                                       'CUDA_VISIBLE_DEVICES': visible}, clear=True), \
             patch.dict('sys.modules', {'torch': torch}):
            require_gpu()

    def test_pipeline_count_rejects_mig_slices(self):
        """Two-device admission still requires full GPU- UUIDs (no slice P2P)."""
        torch = SimpleNamespace(version=SimpleNamespace(cuda='13.0', hip=None),
                                cuda=SimpleNamespace(is_available=lambda: True,
                                                     device_count=lambda: 2,
                                                     set_device=lambda index: None))
        visible = ('MIG-c083756b-37b9-51b5-ac0e-6cdda07b784c,'
                   'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283')
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda',
                                       'CUDA_VISIBLE_DEVICES': visible,
                                       'DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT': '12'},
                        clear=True), \
             patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'distinct GPU UUIDs'):
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
