import unittest
from unittest.mock import patch
from damendaemon.cuda_glm_flash_linear import CudaFlashLinearAttention
from tests.test_glm_flash_linear import layer


class CudaFlashLinearAdmissionTests(unittest.TestCase):
    def test_cpu_weights_rejected_before_core(self):
        cpu=layer()
        with patch('damendaemon.cuda_glm_flash_linear._FlashLinearAttention.__init__',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashLinearAttention(cpu.weights,heads=2,head_dim=3)

    def test_missing_projection_rejected(self):
        for weights in (None,{}, {'q_proj.weight':None}):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashLinearAttention(weights,heads=2,head_dim=3)
