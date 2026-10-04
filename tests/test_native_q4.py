import os
import unittest
from unittest.mock import patch
import torch
from dllm.native_q4 import library,linear_if_enabled


class NativeQ4Tests(unittest.TestCase):
    def tearDown(self):
        library.cache_clear()

    def test_explicit_opt_in_and_shape_fallback(self):
        x=torch.zeros(1,4096,dtype=torch.float16)
        weight=torch.zeros(1024,2048,dtype=torch.uint8)
        scales=torch.ones(1024,128,dtype=torch.float16)
        with patch.dict(os.environ,{'DLLM_Q4_CUDA_LIBRARY':''}):
            self.assertIsNone(linear_if_enabled(x,weight,scales))
        with patch.dict(os.environ,{'DLLM_Q4_CUDA_LIBRARY':'/unused'}),patch('ctypes.CDLL') as load:
            self.assertIsNone(linear_if_enabled(x.expand(2,-1),weight,scales))
            self.assertIsNone(linear_if_enabled(x.float(),weight,scales))
            with self.assertRaisesRegex(ValueError,'buffers'):
                linear_if_enabled(x,weight,scales)
            load.assert_not_called()

    def test_abi_mismatch_rejected_before_kernel_resolution(self):
        # C functions allow argtype/restype attributes; use an ordinary callable.
        handle=type('Library',(),{})()
        handle.dllm_q4_abi_version=lambda:2
        with patch('ctypes.CDLL',return_value=handle),self.assertRaisesRegex(RuntimeError,'ABI'):
            library('/mismatched-library')
