import unittest
import torch
from dllm.cuda_host_stage import CudaHostStageSlots,_host_tensor,_budget


class CudaHostBoundaryTests(unittest.TestCase):
    def test_requires_explicit_cuda_owner_and_bounded_payload(self):
        with self.assertRaises(ValueError):CudaHostStageSlots(object(),max_transfer_bytes=1024)
        for limit in (True,0,-1,2**31+1):
            with self.assertRaises(ValueError):_budget(0,limit)
        with self.assertRaises(ValueError):_budget(33,32)
        _budget(32,32)

    def test_host_geometry_before_transfer(self):
        self.assertEqual(_host_tensor(torch.ones(2,3),torch.float32,(None,3)),24)
        for value in (torch.ones(2,4),torch.ones(2,3).half(),torch.ones(0,3),torch.ones(2,3,1)):
            with self.assertRaises(ValueError):_host_tensor(value,torch.float32,(None,3))
