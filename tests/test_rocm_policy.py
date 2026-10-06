"""ROCm selection/admission tests; no mocked GPU correctness or speed claims."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from dllm.device import selected_accelerator,selected_device,available_memory
from dllm.rocm_policy import NVIDIA_LIBRARIES,validate_runtime,memory_limit,prefer_q8_small_batch_matvec,configure_attention,configure_allocator,require_host_headroom
from dllm.runtime import require_gpu

GIB=1024**3
ENV={'DLLM_DEVICE':'rocm','ROCR_VISIBLE_DEVICES':'0','HIP_VISIBLE_DEVICES':'0'}


class RocmPolicyTests(unittest.TestCase):
    def test_attention_default_and_explicit_diagnostic_override(self):
        import os
        key = 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'
        with patch.dict(os.environ, {}, clear=True):
            configure_attention()
            self.assertEqual(os.environ[key], '1')
        with patch.dict(os.environ, {key: '0'}, clear=True):
            configure_attention()
            self.assertEqual(os.environ[key], '0')
        with patch.dict(os.environ, {key: 'invalid'}, clear=True):
            with self.assertRaises(ValueError):
                configure_attention()

    def test_q8_policy_is_limited_to_qualified_architecture_and_rows(self):
        torch = self.torch()
        for rows in (1, 2, 3, 4, 5, 128):
            self.assertEqual(prefer_q8_small_batch_matvec(torch, 0, rows), 2 <= rows <= 4)
        for architecture in ('gfx1100', 'gfx942', 'gfx11510'):
            torch.cuda.get_device_properties = lambda device: SimpleNamespace(gcnArchName=architecture)
            self.assertFalse(prefer_q8_small_batch_matvec(torch, 0, 4))
        torch.cuda.get_device_properties = lambda device: SimpleNamespace(gcnArchName='gfx1151:xnack-')
        self.assertTrue(prefer_q8_small_batch_matvec(torch, 0, 4))

    def test_q8_policy_does_not_query_nvidia_architecture(self):
        torch = self.torch()
        torch.version.hip = None
        torch.cuda.get_device_properties = Mock(side_effect=AssertionError('NVIDIA must keep existing dispatch'))
        self.assertFalse(prefer_q8_small_batch_matvec(torch, 0, 4))
        torch.cuda.get_device_properties.assert_not_called()

    def torch(self):
        return SimpleNamespace(version=SimpleNamespace(hip='7.13'),cuda=SimpleNamespace(
            is_available=lambda:True,device_count=lambda:1,set_device=Mock(),
            get_device_properties=lambda index:SimpleNamespace(gcnArchName='gfx1151'),
            mem_get_info=lambda:(111*GIB,61*GIB),memory_allocated=lambda:2*GIB,
            memory_reserved=lambda:2*GIB,get_per_process_memory_fraction=lambda device:1.0,
            set_per_process_memory_fraction=Mock()))

    def test_explicit_hip_selection_uses_cuda_tensor_namespace(self):
        torch=self.torch()
        with patch.dict('os.environ',ENV,clear=True),patch.dict('sys.modules',{'torch':torch}),patch('dllm.rocm_policy.host_available',return_value=100*GIB):
            require_gpu()
            self.assertEqual(selected_device(),'cuda')
            self.assertEqual(selected_accelerator(),'rocm')
            torch.cuda.set_device.assert_called_once_with(0)
            torch.cuda.set_per_process_memory_fraction.assert_called_once_with(8/61, 0)

    def test_allocator_preserves_stricter_budget_and_counts_existing_reservations(self):
        torch = self.torch()
        torch.cuda.get_per_process_memory_fraction = lambda device: .05
        with patch.dict('os.environ',ENV,clear=True),patch('dllm.rocm_policy.host_available',return_value=100*GIB):
            self.assertEqual(configure_allocator(torch), int(61*GIB*.05))
            self.assertAlmostEqual(torch.cuda.set_per_process_memory_fraction.call_args.args[0], .05)
        torch.cuda.set_per_process_memory_fraction.reset_mock()
        torch.cuda.memory_reserved = lambda: 9*GIB
        with patch.dict('os.environ',ENV,clear=True),patch('dllm.rocm_policy.host_available',return_value=100*GIB),self.assertRaises(MemoryError):
            configure_allocator(torch)
        torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_low_host_headroom_and_failed_enforcement_fail_closed(self):
        torch = self.torch()
        with patch('dllm.rocm_policy.host_available',return_value=16*GIB),self.assertRaises(MemoryError):
            require_host_headroom()
        torch.cuda.set_per_process_memory_fraction.side_effect = RuntimeError('unsupported allocator')
        with patch.dict('os.environ',ENV,clear=True),patch('dllm.rocm_policy.host_available',return_value=100*GIB),self.assertRaisesRegex(RuntimeError,'unsupported allocator'):
            configure_allocator(torch)

    def test_mixed_visibility_and_native_libraries_reject_before_selection(self):
        cases=[{'CUDA_VISIBLE_DEVICES':'GPU-a'},{'ROCR_VISIBLE_DEVICES':'0,1'},
               {'ROCR_VISIBLE_DEVICES':'０'},{'HIP_VISIBLE_DEVICES':'1'},
               {'DLLM_EXPERIMENTAL_LAYER_SPLIT':'12'}]
        cases += [{name:'/tmp/nvidia-only.so'} for name in NVIDIA_LIBRARIES]
        for values in cases:
            with self.subTest(values=values),patch.dict('os.environ',ENV|values,clear=True):
                with self.assertRaises(ValueError):validate_runtime(self.torch())

    def test_wrong_build_device_count_and_architecture_reject(self):
        for kind in ('build','count','architecture'):
            torch=self.torch()
            if kind=='build':torch.version.hip=None
            if kind=='count':torch.cuda.device_count=lambda:2
            if kind=='architecture':torch.cuda.get_device_properties=lambda index:SimpleNamespace(gcnArchName='gfx9999')
            with self.subTest(kind=kind),patch.dict('os.environ',ENV,clear=True):
                with self.assertRaises((ValueError,RuntimeError)):validate_runtime(torch)

    def test_budget_counts_live_allocations_and_reserves_host_headroom(self):
        torch=self.torch()
        with patch.dict('os.environ',ENV,clear=True),patch.dict('sys.modules',{'torch':torch}),patch('dllm.rocm_policy.host_available',return_value=100*GIB):
            self.assertEqual(available_memory(),6*GIB)
        with patch.dict('os.environ',ENV,clear=True),patch.dict('sys.modules',{'torch':torch}),patch('dllm.rocm_policy.host_available',return_value=17*GIB):
            self.assertEqual(available_memory(),GIB)
        with patch.dict('os.environ',ENV,clear=True),patch.dict('sys.modules',{'torch':torch}),patch('dllm.rocm_policy.host_available',return_value=15*GIB):
            self.assertEqual(available_memory(),0)

    def test_inconsistent_free_above_total_cannot_raise_admission_above_total(self):
        torch=self.torch()
        with patch.dict('os.environ',ENV|{'DLLM_ROCM_MEMORY_GIB':'120'},clear=True),patch.dict('sys.modules',{'torch':torch}),patch('dllm.rocm_policy.host_available',return_value=120*GIB):
            self.assertEqual(available_memory(),61*GIB)

    def test_invalid_memory_limit_rejects(self):
        for value in ('0','-1','nan','inf','1e308','bad'):
            with self.subTest(value=value),patch.dict('os.environ',{'DLLM_ROCM_MEMORY_GIB':value},clear=True):
                with self.assertRaises(ValueError):memory_limit()

    def test_hip_build_cannot_bypass_apu_budget_using_cuda_selection(self):
        torch=self.torch()
        with patch.dict('os.environ',{'DLLM_DEVICE':'cuda','CUDA_VISIBLE_DEVICES':'GPU-example'},clear=True),patch.dict('sys.modules',{'torch':torch}):
            with self.assertRaisesRegex(RuntimeError,'explicit'):require_gpu()
            with self.assertRaisesRegex(RuntimeError,'explicit'):available_memory()
            torch.cuda.set_device.assert_not_called()
