import unittest
from unittest.mock import patch
import torch
from dllm.cuda_packed_expert import CudaPackedExpert
from dllm.cuda_packed_moe import cuda_packed_moe, _accumulate


class CudaPackedMoeTests(unittest.TestCase):
    def bank(self,device):
        bank=[]
        for multiplier in (2.,3.):
            expert=CudaPackedExpert.__new__(CudaPackedExpert)
            expert.device=torch.device(device);expert.width=2
            expert._row=lambda x,c,m=multiplier:(x.float()*m*(1 if c is None else c)).to(x.dtype)
            bank.append(expert)
        return bank

    def test_cpu_composition_control_duplicates_and_shared(self):
        bank=self.bank('cpu');x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16)
        ids=torch.tensor([[1,1],[1,0]]);weights=torch.tensor([[.25,.5],[0.,.75]])
        result=_accumulate(x,bank,ids,weights,bank[0])
        expected=torch.stack([x[0]*4.25,x[1]*3.5])
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        self.assertEqual(_accumulate(x[:0],bank,ids[:0],weights[:0],bank[0]).shape,(0,2))
        with self.assertRaises(ValueError):cuda_packed_moe(x,bank,ids,weights)

    def test_fake_cuda_metadata_rejections(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            bank=self.bank('cuda:0');x=torch.empty(0,2,device='cuda:0')
            ids=torch.empty(0,2,device='cuda:0',dtype=torch.int64);weights=torch.empty(0,2,device='cuda:0')
            with patch('dllm.cuda_packed_moe._accumulate',return_value=x) as compute:
                self.assertIs(cuda_packed_moe(x,bank,ids,weights),x)
                compute.reset_mock()
                for bad in (torch.empty(0,2),torch.empty(0,3,device='cuda:0')):
                    with self.assertRaises(ValueError):cuda_packed_moe(bad,bank,ids,weights)
                with self.assertRaises(ValueError):cuda_packed_moe(x,bank,ids.to(torch.int32),weights)
                with self.assertRaises(ValueError):cuda_packed_moe(x,bank,ids,weights.half())
                with self.assertRaises(ValueError):cuda_packed_moe(x,bank,ids,weights,shared=self.bank('cuda:1')[0])
                compute.assert_not_called()

    def test_invalid_route_rejected_before_group_allocation(self):
        # CPU control of value validation, explicitly bypassing device admission.
        bank=self.bank('cpu');x=torch.ones(1,2);weights=torch.ones(1,1)
        with patch('dllm.cuda_packed_moe._validate'), patch('dllm.cuda_packed_moe._accumulate') as compute:
            for value in (-1,2,2**60):
                with self.assertRaises(ValueError):cuda_packed_moe(x,bank,torch.tensor([[value]]),weights)
            compute.assert_not_called()


if __name__=='__main__':unittest.main()
