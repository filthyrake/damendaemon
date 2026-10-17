import unittest
from unittest.mock import patch
from damendaemon.cuda_glm_flash_sparse import CudaFlashSparseAttention
from tests.test_glm_flash_sparse import fixture


class CudaFlashSparseAdmissionTests(unittest.TestCase):
    def test_cpu_indexer_rejected_before_core(self):
        cpu,_,_=fixture()
        with patch('damendaemon.cuda_glm_flash_sparse._FlashSparseAttention.__init__',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA indexer'):CudaFlashSparseAttention(cpu.weights,cpu.indexer,heads=2,head_dim=3,value_dim=2)

    def test_missing_indexer_rejected(self):
        with self.assertRaisesRegex(ValueError,'CUDA indexer'):CudaFlashSparseAttention({},None,heads=2,head_dim=3,value_dim=2)
