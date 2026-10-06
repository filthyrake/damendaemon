import os
import unittest
from unittest.mock import Mock, patch

import torch

from dllm.native_q4_marlin import Library, select
from dllm.q4_marlin_policy import LayoutPlan
from dllm.quantization import Linear


class NativeQ4MarlinTests(unittest.TestCase):
    def test_abi_rejection_precedes_device_init(self):
        for abi in (0,2,201,202):
            library=Mock()
            library.dllm_q4_marlin_abi.return_value=abi
            with patch('ctypes.CDLL',return_value=library):
                with self.assertRaisesRegex(ValueError,'ABI'):Library('/test')
            library.dllm_q4_marlin_init.assert_not_called()

    def test_selection_allocates_no_tensors_and_preserves_library_identity(self):
        def initialize(sms,shared):
            sms._obj.value=10
            shared._obj.value=101376
            return 0
        libraries=[]
        for _ in range(2):
            library=Mock()
            library.dllm_q4_marlin_abi.return_value=1
            library.dllm_q4_marlin_init.side_effect=initialize
            libraries.append(library)
        with patch('ctypes.CDLL',side_effect=libraries), \
             patch('torch.empty',side_effect=AssertionError('allocated before admission')), \
             patch('torch.zeros',side_effect=AssertionError('allocated before admission')):
            first,second=Library('/first'),Library('/second')
        self.assertIs(first.lib,libraries[0])
        self.assertIs(second.lib,libraries[1])
        self.assertIsNone(first.scratch)
        self.assertEqual(first.workspace_bytes,2621600)
        with self.assertRaisesRegex(ValueError,'activation'):
            first.linear(torch.ones(1,128,dtype=torch.float16),Mock())

    def test_disabled_or_unqualified_selection_does_not_load_library(self):
        args=({}, {}, {}, 'q4_0','cuda',torch.float16,'mixed',512,4,True)
        with patch.dict(os.environ,{},clear=True), \
             patch('torch.cuda.get_device_name',side_effect=AssertionError('queried CUDA')):
            self.assertEqual(select(*args),(None,LayoutPlan()))
        with patch.dict(os.environ,{'DLLM_Q4_MARLIN_LIBRARY':'/invalid'}), \
             patch('torch.cuda.get_device_name',return_value='NVIDIA A2'), \
             patch('ctypes.CDLL',side_effect=AssertionError('loaded library')):
            self.assertEqual(select(*args),(None,LayoutPlan()))

    def test_prepared_dispatch_cannot_fall_through_to_dense_or_canonical_q4(self):
        weight=torch.empty(8,8,dtype=torch.int32)
        backend=Mock()
        backend.linear.return_value=torch.zeros(6,8,dtype=torch.float16)
        linear=Linear({}, {}, 'cpu',Mock(side_effect=AssertionError('fell through')))
        linear.native_q4_marlin=backend
        matrix=Mock(outputs=8)
        linear.q4_marlin[id(weight)]=matrix
        x=torch.ones(2,3,16,dtype=torch.float16)
        self.assertEqual(linear(x,weight).shape,(2,3,8))
        self.assertIs(backend.linear.call_args.args[1],matrix)
        with self.assertRaisesRegex(ValueError,'bias'):linear(x,weight,torch.zeros(8))


if __name__=='__main__':unittest.main()
