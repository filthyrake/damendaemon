import unittest
from unittest.mock import patch
from damendaemon.cuda_hyper_connections import reduce_streams,reduce_head,restore_streams
from tests import test_hyper_connections


class CudaHyperAdmissionTests(unittest.TestCase):
    def test_cpu_rejected_before_core(self):
        hidden,projection,scale,bias=test_hyper_connections.HyperConnectionTests().fixture()
        with patch('damendaemon.cuda_hyper_connections.core.reduce_streams',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA'):reduce_streams(hidden,projection,scale,bias)
        with self.assertRaisesRegex(ValueError,'CUDA'):reduce_head(hidden,projection[:4],scale[:1],bias[:4])
        with self.assertRaisesRegex(ValueError,'CUDA'):restore_streams(hidden[:,:,0],hidden,None,None)

    def test_missing_inputs_fail_closed(self):
        with self.assertRaises(ValueError):reduce_streams(None,None,None,None)
        with self.assertRaises(ValueError):reduce_head(None,None,None,None)
        with self.assertRaises(ValueError):restore_streams(None,None,None,None)
