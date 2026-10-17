import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_selected_attention import selected_attention


class CudaSelectedAdmissionTests(unittest.TestCase):
    def test_cpu_rejected_before_core(self):
        with patch('damendaemon.cuda_selected_attention._selected_attention',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA'):selected_attention(torch.zeros(1,1,1,2),torch.zeros(1,1,2),torch.zeros(1,1,1,dtype=torch.int64),torch.zeros(1))

    def test_missing_query_rejected(self):
        with self.assertRaises(ValueError):selected_attention(None,None,None,None)
