import unittest
from unittest.mock import patch
import torch
from dllm.cuda_compressed_pool import pool_projected,reorder_pool,_budget
from dllm.compressed_pool import pool_projected as cpu_pool
from tests.test_compressed_pool import CompressedPoolTests


class CudaPoolAdmissionTests(unittest.TestCase):
    def test_cpu_inputs_rejected_before_core(self):
        values,gates,offsets=CompressedPoolTests().fixture(4)
        with patch('dllm.cuda_compressed_pool._pool',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'same-device'):pool_projected(values,gates,offsets)
        _,state,_=cpu_pool(values,gates,offsets)
        with self.assertRaisesRegex(ValueError,'state'):reorder_pool(state,torch.tensor([0]))

    def test_exact_budget_boundary_and_invalid_limits(self):
        _budget(100,50,100,50)
        for args in ((101,50,100,50),(100,51,100,50),(1,1,True,1),(1,1,0,1),(1,1,1,2**31+1)):
            with self.assertRaisesRegex(ValueError,'budget'):_budget(*args)

    def test_missing_inputs_fail_closed(self):
        with self.assertRaises(ValueError):pool_projected(None,None,None)
        with self.assertRaises(ValueError):reorder_pool(None,None)
