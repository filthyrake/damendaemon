"""ROCm selection/admission tests; no mocked GPU correctness or speed claims."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from dllm.device import selected_accelerator,selected_device,available_memory
from dllm.rocm_policy import NVIDIA_LIBRARIES,validate_runtime,memory_limit
from dllm.runtime import require_gpu

GIB=1024**3
ENV={'DLLM_DEVICE':'rocm','ROCR_VISIBLE_DEVICES':'0','HIP_VISIBLE_DEVICES':'0'}


class RocmPolicyTests(unittest.TestCase):
    def torch(self):
        return SimpleNamespace(version=SimpleNamespace(hip='7.13'),cuda=SimpleNamespace(
            is_available=lambda:True,device_count=lambda:1,set_device=Mock(),
            get_device_properties=lambda index:SimpleNamespace(gcnArchName='gfx1151'),
            mem_get_info=lambda:(111*GIB,61*GIB),memory_allocated=lambda:2*GIB))

    def test_explicit_hip_selection_uses_cuda_tensor_namespace(self):
        torch=self.torch()
        with patch.dict('os.environ',ENV,clear=True),patch.dict('sys.modules',{'torch':torch}):
            require_gpu()
            self.assertEqual(selected_device(),'cuda')
            self.assertEqual(selected_accelerator(),'rocm')
            torch.cuda.set_device.assert_called_once_with(0)

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
