import unittest
from unittest.mock import patch
from damendaemon.cuda_floating_compressor import compress_hidden
from tests import test_floating_compressor


class CudaCompressorAdmissionTests(unittest.TestCase):
    def test_cpu_rejected_before_core(self):
        args=test_floating_compressor.FloatingCompressorTests().fixture()
        with patch('damendaemon.cuda_floating_compressor._compress_hidden',side_effect=AssertionError('entered core')):
            with self.assertRaisesRegex(ValueError,'CUDA'):compress_hidden(*args)

    def test_missing_inputs_rejected(self):
        with self.assertRaises(ValueError):compress_hidden(None,None,None,None,None,None)
