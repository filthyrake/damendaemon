"""Host-side admission checks; numerical CUDA proof lives in the guarded runner."""
import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_glm_flash_primitives import recurrent_attention, CudaFlashHyperConnection
from tests.test_glm_flash_kda import inputs
from tests.test_glm_flash_hyper import component


class CudaFlashAdmissionTests(unittest.TestCase):
    def test_cpu_recurrence_rejected_before_core(self):
        with patch('damendaemon.cuda_glm_flash_primitives._recurrent_attention',side_effect=AssertionError('entered core')):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):recurrent_attention(*inputs())

    def test_cpu_hyper_rejected_before_core(self):
        c=component()
        with patch('damendaemon.cuda_glm_flash_primitives._FlashHyperConnection.__init__',side_effect=AssertionError('entered core')):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashHyperConnection(c.projection,c.base,c.scale,streams=2)

    def test_non_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError,'CUDA tensors'):recurrent_attention(None,None,None,None,None)
        with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashHyperConnection(None,None,None)
