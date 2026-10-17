import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_packed_placement import _plan, place_cuda_experts
from tests.test_packed_moe import expert


class CudaPlacementTests(unittest.TestCase):
    def test_complete_bank_admitted_before_any_copy(self):
        a,_=expert('e2m1');b,_=expert('e4m3fn')
        _,_,needed=_plan([a,b],None,10**6)
        with patch('damendaemon.cuda_packed_placement._copy',side_effect=AssertionError('copied')) as copy:
            with self.assertRaises(ValueError):place_cuda_experts([a,b],device='cuda:0',max_bytes=needed-1)
            b.options['column_chunk']=0
            with self.assertRaises(ValueError):place_cuda_experts([a,b],device='cuda:0',max_bytes=10**6)
            for device in ('cpu','cuda'):
                with self.assertRaises(ValueError):place_cuda_experts([a],device=device,max_bytes=10**6)
            copy.assert_not_called()

    def test_alias_accounting_and_prepared_identity(self):
        a,_=expert('e2m1')
        _,tensors,needed=_plan([a,a],a,10**6)
        self.assertEqual(needed,sum(t.numel()*t.element_size() for t in tensors.values()))
        # CPU copies/constructor substitute verify ownership graph only.
        with patch('damendaemon.cuda_packed_placement._copy',side_effect=lambda t,d:t.clone()) as copy, \
             patch('damendaemon.cuda_packed_placement.CudaPackedExpert',side_effect=lambda *m,**kw:(m,kw)):
            bank=place_cuda_experts([a,a],shared=a,device='cuda:0',max_bytes=needed)
        self.assertEqual(copy.call_count,len(tensors))
        self.assertIs(bank.experts[0],bank.experts[1]);self.assertIs(bank.shared,bank.experts[0])
        self.assertEqual(bank.raw_bytes,needed)
        copied=bank.experts[0][0][0].weight
        self.assertNotEqual(copied.data_ptr(),a.gate.weight.data_ptr())
        torch.testing.assert_close(copied,a.gate.weight)

    def test_copy_failure_does_not_construct_partial_bank(self):
        a,_=expert('e2m1');snap=a.gate.weight.clone()
        with patch('damendaemon.cuda_packed_placement._copy',side_effect=[snap,RuntimeError('allocation')]), \
             patch('damendaemon.cuda_packed_placement.CudaPackedExpert') as construct:
            with self.assertRaisesRegex(RuntimeError,'allocation'):
                place_cuda_experts([a],device='cuda:0',max_bytes=10**6)
            construct.assert_not_called()
        torch.testing.assert_close(a.gate.weight,snap)


if __name__=='__main__':unittest.main()
