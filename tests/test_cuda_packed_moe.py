import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_packed_expert import CudaPackedExpert
from damendaemon.cuda_packed_moe import _accumulate, _accumulate_batched, cuda_packed_moe


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

    def batched_bank(self,device):
        # Fakes exposing both interfaces; forward_batched mirrors _row per row.
        bank=self.bank(device)
        for expert in bank:
            def make_forward(e):
                def forward_batched(hidden,*,route_weights=None):
                    output=torch.empty_like(hidden)
                    for row in range(hidden.shape[0]):
                        coefficient=None if route_weights is None else route_weights[row:row+1]
                        output[row:row+1]=e._row(hidden[row:row+1],coefficient)
                    return output
                return forward_batched
            expert.forward_batched=make_forward(expert)
        return bank

    def test_batched_composition_matches_control_exactly_on_cpu(self):
        bank=self.batched_bank('cpu')
        rng=torch.Generator().manual_seed(479)
        cases=[(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16),
                torch.tensor([[1,1],[1,0]]),torch.tensor([[.25,.5],[0.,.75]])),
               (torch.randn(3,2,generator=rng).half(),
                torch.tensor([[0,1],[1,1],[1,1]]),torch.rand(3,2,generator=rng)),
               (torch.randn(2,2,generator=rng).half(),
                torch.zeros(2,2,dtype=torch.int64),torch.ones(2,2)),
               (torch.zeros(0,2,dtype=torch.float16),
                torch.zeros(0,2,dtype=torch.int64),torch.zeros(0,2))]
        for hidden,ids,weights in cases:
            for shared in (None,bank[0]):
                with self.subTest(shared=shared is not None,rows=hidden.shape[0]):
                    control=_accumulate(hidden,bank,ids,weights,shared)
                    batched=_accumulate_batched(hidden,bank,ids,weights,shared)
                    torch.testing.assert_close(batched,control,rtol=0,atol=0)

    def test_batched_opt_in_requires_forward_batched_and_validates(self):
        bank=self.bank('cpu');x=torch.tensor([[1.,2.]],dtype=torch.float16)
        ids=torch.tensor([[1,0]]);weights=torch.tensor([[.5,.5]])
        with self.assertRaises(ValueError):cuda_packed_moe(x,bank,ids,weights,batched=True)
        real=self.batched_bank('cpu')
        with patch('damendaemon.cuda_packed_moe._validate'), \
                patch('damendaemon.cuda_packed_moe._accumulate_batched',return_value=x) as compute:
            cuda_packed_moe(x,real,ids,weights,batched=True)
            compute.assert_called_once()
            compute.reset_mock()
            cuda_packed_moe(x,real,ids,weights)
            compute.assert_not_called()

    def test_fake_cuda_metadata_rejections(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            bank=self.bank('cuda:0');x=torch.empty(0,2,device='cuda:0')
            ids=torch.empty(0,2,device='cuda:0',dtype=torch.int64);weights=torch.empty(0,2,device='cuda:0')
            with patch('damendaemon.cuda_packed_moe._accumulate',return_value=x) as compute:
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
        with patch('damendaemon.cuda_packed_moe._validate'), patch('damendaemon.cuda_packed_moe._accumulate') as compute:
            for value in (-1,2,2**60):
                with self.assertRaises(ValueError):cuda_packed_moe(x,bank,torch.tensor([[value]]),weights)
            compute.assert_not_called()


if __name__=='__main__':unittest.main()
