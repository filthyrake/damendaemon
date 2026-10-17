import unittest
import torch
from types import SimpleNamespace
from damendaemon.cuda_deepseek_block import CudaDeepSeekBlock,_Backend


class CudaBlockAdmissionTests(unittest.TestCase):
    def test_unprepared_block_rejected(self):
        with self.assertRaises(ValueError):
            CudaDeepSeekBlock({}, {},None,packed_experts=[],packed_shared=None,groups=2,window=4)

    def test_whole_block_budget_precedes_value_scan(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from unittest.mock import patch
        device=torch.device('cuda:0')
        backend=_Backend(SimpleNamespace(device=device),1024,1024)
        block=SimpleNamespace(hc_mult=2,dim=8,table=None,weights={})
        with FakeTensorMode(),patch('damendaemon.cuda_deepseek_block.finite',side_effect=AssertionError('scan before admission')):
            with self.assertRaises(ValueError):backend.admit(torch.empty(1,3,2,8,device=device),None,block)
