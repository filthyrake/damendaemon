import unittest
from unittest.mock import patch
from damendaemon.cuda_compressed_indexer import select_compressed
from tests import test_compressed_indexer


class CudaIndexerAdmissionTests(unittest.TestCase):
    def test_cpu_inputs_rejected_before_core(self):
        args=test_compressed_indexer.CompressedIndexerTests().fixture()
        with patch('damendaemon.cuda_compressed_indexer._select_compressed',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA'):select_compressed(*args,ratio=4,top_k=5)

    def test_missing_query_rejected(self):
        with self.assertRaises(ValueError):select_compressed(None,None,None,None,ratio=4,top_k=5)
