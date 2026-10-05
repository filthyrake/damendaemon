import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from dllm.awq_decode_policy import _is_a2, use_small_rows


class AwqDecodePolicyTests(unittest.TestCase):
    def tearDown(self):
        _is_a2.cache_clear()

    def test_unqualified_rows_shapes_and_devices_do_not_query_cuda(self):
        with patch('torch.cuda.get_device_properties', side_effect=AssertionError('unexpected device query')):
            for rows, inputs, outputs in [(0, 896, 896), (3, 896, 1152), (4, 896, 9728),
                                           (5, 4864, 896), (1, 1024, 1024), (129, 896, 896)]:
                self.assertFalse(use_small_rows(rows, inputs, outputs, torch.device('cuda:0')))
            self.assertFalse(use_small_rows(1, 896, 896, torch.device('mps')))

    def test_measured_shapes_and_gpu_cache_are_isolated(self):
        def properties(device):
            return SimpleNamespace(name='NVIDIA A2' if device.index == 0 else 'NVIDIA A30')
        with patch('torch.cuda.get_device_properties', side_effect=properties) as query:
            for inputs, outputs, maximum in [(896, 1152, 2), (896, 896, 4), (896, 9728, 3), (4864, 896, 4)]:
                for rows in range(1, maximum+1):
                    self.assertTrue(use_small_rows(rows, inputs, outputs, torch.device('cuda:0')))
                    self.assertFalse(use_small_rows(rows, inputs, outputs, torch.device('cuda:1')))
                self.assertFalse(use_small_rows(maximum+1, inputs, outputs, torch.device('cuda:0')))
            self.assertEqual(query.call_count, 2)
