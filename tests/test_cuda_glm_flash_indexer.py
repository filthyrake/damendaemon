import unittest
from unittest.mock import patch
from dllm.cuda_glm_flash_indexer import CudaFlashIndexer
from tests.test_glm_flash_indexer import indexer


class CudaFlashIndexerAdmissionTests(unittest.TestCase):
    def test_cpu_weights_rejected_before_core(self):
        cpu=indexer()
        with patch('dllm.cuda_glm_flash_indexer._FlashIndexer.__init__',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashIndexer(cpu.weights,heads=2,pool_size=2,top_k=2)

    def test_missing_projection_rejected(self):
        for weights in (None,{}, {'wk.weight':None}):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashIndexer(weights,heads=2)
