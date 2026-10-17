import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_glm_flash_vision import CudaFlashVision
from tests.test_glm_flash_vision import fixture


class CudaFlashVisionAdmissionTests(unittest.TestCase):
    def test_cpu_weights_rejected(self):
        with self.assertRaises(ValueError):CudaFlashVision(*fixture())

    def test_budget_and_device_rejected_before_projection(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        config,weights=fixture()
        with FakeTensorMode():
            weights={name:torch.empty(value.shape,device='cuda',dtype=value.dtype) for name,value in weights.items()}
            with self.assertRaises(ValueError):CudaFlashVision(config,weights,max_weight_bytes=1)
            encoder=CudaFlashVision(config,weights,max_workspace_bytes=1)
            with patch('damendaemon.glm_flash_vision.F.linear',side_effect=AssertionError('projection before admission')):
                with self.assertRaises(ValueError):encoder.forward(torch.empty(4,24,device='cuda'),[(1,2,2)])
                with self.assertRaises(ValueError):encoder.forward(torch.empty(4,24),[(1,2,2)])
