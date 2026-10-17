import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from damendaemon.q4_prefill_policy import _is_a2, use_small_tile


class Q4PrefillPolicyTests(unittest.TestCase):
    def tearDown(self):
        _is_a2.cache_clear()

    def test_decode_and_unqualified_devices_never_query_cuda(self):
        with patch('torch.cuda.get_device_properties', side_effect=AssertionError('unexpected CUDA lookup')):
            for rows in (1, 2, 4, 16, 32, 65, 128, 129, 160):
                self.assertFalse(use_small_tile(rows, 6144, 4096, torch.device('cuda:0')))
            self.assertFalse(use_small_tile(64, 6144, 4096, torch.device('cpu')))
            self.assertFalse(use_small_tile(64, 4096, 2048, torch.device('cuda:0')))

    def test_hardware_profiles_are_cached_and_isolated_per_device(self):
        def properties(device):
            return SimpleNamespace(name='NVIDIA A2' if device.index == 0 else 'NVIDIA A30')
        with patch('torch.cuda.get_device_properties', side_effect=properties) as query:
            for rows in (33, 48, 64):
                self.assertTrue(use_small_tile(rows, 6144, 4096, torch.device('cuda:0')))
                self.assertFalse(use_small_tile(rows, 6144, 4096, torch.device('cuda:1')))
            self.assertEqual(query.call_count, 2)
