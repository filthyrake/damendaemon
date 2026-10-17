import os
import unittest
from unittest.mock import Mock, patch

import torch

from damendaemon.native_q4_marlin import Library, select
from damendaemon.q4_marlin_policy import LayoutPlan
from damendaemon.quantization import Linear


class NativeQ4MarlinTests(unittest.TestCase):
    def test_abi_rejection_precedes_device_init(self):
        for abi in (0,3,201,202):
            library=Mock()
            library.damendaemon_q4_marlin_abi.return_value=abi
            with patch('ctypes.CDLL',return_value=library):
                with self.assertRaisesRegex(ValueError,'ABI'):Library('/test')
            library.damendaemon_q4_marlin_init.assert_not_called()

    def test_selection_allocates_no_tensors_and_preserves_library_identity(self):
        def initialize(sms,shared):
            sms._obj.value=10
            shared._obj.value=101376
            return 0
        libraries=[]
        for abi in (1,2):
            library=Mock()
            library.damendaemon_q4_marlin_abi.return_value=abi
            library.damendaemon_q4_marlin_init.side_effect=initialize
            libraries.append(library)
        with patch('ctypes.CDLL',side_effect=libraries), \
             patch('torch.empty',side_effect=AssertionError('allocated before admission')), \
             patch('torch.zeros',side_effect=AssertionError('allocated before admission')):
            first,second=Library('/first'),Library('/second')
        self.assertIs(first.lib,libraries[0])
        self.assertIs(second.lib,libraries[1])
        self.assertEqual((first.abi,second.abi),(1,2))
        self.assertIsNone(first.scratch)
        self.assertEqual(first.workspace_bytes,2621600)
        with self.assertRaisesRegex(ValueError,'activation'):
                first.linear(torch.ones(1,128,dtype=torch.float16),Mock())

    def test_selected_library_abi_controls_admitted_layout_bytes(self):
        original=LayoutPlan(('projection',),100,120,20,50)
        args=({}, {}, {}, 'q4_0','cuda',torch.float16,'mixed',512,4,True)
        for abi in (1,2):
            library=Mock(abi=abi)
            with patch.dict(os.environ,{'DAMENDAEMON_Q4_MARLIN_LIBRARY':'/qualified'}), \
                 patch('torch.cuda.get_device_name',return_value='NVIDIA A2'), \
                 patch('damendaemon.native_q4_marlin.plan',return_value=original), \
                 patch('damendaemon.native_q4_marlin.Library',return_value=library):
                actual,selected=select(*args)
            self.assertIs(actual,library)
            self.assertEqual(selected.overhead_bytes,20 if abi==1 else 0)
            self.assertEqual(selected.prepared_bytes,120 if abi==1 else 100)
            self.assertEqual(selected.repack_peak_bytes,50)

    def test_disabled_or_unqualified_selection_does_not_load_library(self):
        args=({}, {}, {}, 'q4_0','cuda',torch.float16,'mixed',512,4,True)
        with patch.dict(os.environ,{},clear=True), \
             patch('torch.cuda.get_device_name',side_effect=AssertionError('queried CUDA')):
            self.assertEqual(select(*args),(None,LayoutPlan()))
        with patch.dict(os.environ,{'DAMENDAEMON_Q4_MARLIN_LIBRARY':'/invalid'}), \
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
