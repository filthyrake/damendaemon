import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_floating_indexer import index_hidden
from tests import test_floating_indexer


class CudaLearnedIndexerAdmissionTests(unittest.TestCase):
    def test_cpu_rejected_before_core(self):
        args=test_floating_indexer.FloatingIndexerTests().fixture(torch.float32)
        with patch('damendaemon.cuda_floating_indexer._index_hidden',side_effect=AssertionError('entered core')):
            with self.assertRaisesRegex(ValueError,'CUDA'):index_hidden(*args,top_k=3)

    def test_missing_inputs_rejected(self):
        with self.assertRaises(ValueError):index_hidden(None,None,None,None,None,None,None,top_k=3)
