import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from dllm.cuda_glm_ffn import CudaPackedGlmFeedForward,_select,_routed,_routed_batched
from dllm.cuda_packed_expert import CudaPackedExpert
from dllm.glm_ffn import select_glm_experts


class CudaGlmTests(unittest.TestCase):
    def test_routing_underflow_bias_and_ties_match_cpu_control(self):
        logits=torch.tensor([[-1000.,-1000.,-1000.],[-46.,-47.,-48.],[1.,1.,1.]])
        bias=torch.tensor([1.,0.,0.])
        actual,ids=_select(logits,bias,2,2.5)
        expected,reference_ids=select_glm_experts(logits,bias,2)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(ids,reference_ids,rtol=0,atol=0)
        self.assertEqual(actual[0].sum().item(),0)
        self.assertLess(actual[1].sum().item(),2.5)

    def test_device_routing_select_matches_host_path_exactly(self):
        logits=torch.tensor([[-1000.,-1000.,-1000.],[-46.,-47.,-48.],[1.,1.,1.],[2.,2.,1.]])
        bias=torch.tensor([1.,0.,0.])
        host_weights,host_ids=_select(logits,bias,2,2.5)
        device_weights,device_ids=_select(logits,bias,2,2.5,device_routing=True)
        torch.testing.assert_close(device_weights,host_weights,rtol=0,atol=0)
        self.assertTrue(torch.equal(device_ids,host_ids))
        self.assertEqual(device_weights.device,logits.device)

    def test_batched_composition_matches_per_token_control(self):
        # Fake experts exposing forward/forward_batched; the batched path runs
        # all rows per expert and weights AFTER the down projection.
        rng=torch.Generator().manual_seed(484)
        experts=[SimpleNamespace(forward=Mock(side_effect=lambda x,i=i:x*(3+i)),
                                 forward_batched=Mock(side_effect=lambda x,i=i:x*(3+i)))
                 for i in range(3)]
        shared=SimpleNamespace(forward=Mock(side_effect=lambda x:x*7),
                               forward_batched=Mock(side_effect=lambda x:x*7))
        logits=torch.randn(6,3,generator=rng)
        coefficients,ids=_select(logits,torch.zeros(3),2,2.5)
        rows=torch.randn(6,4,generator=rng)
        per_token=_routed(rows,experts,ids,coefficients,shared)
        batched=_routed_batched(rows,experts,ids,coefficients,shared)
        torch.testing.assert_close(batched,per_token,rtol=0,atol=0)
        # Coefficients touched only the completed down projection.
        for expert in experts:
            for call in expert.forward_batched.call_args_list:
                self.assertNotIn('route_weights',call.kwargs)
        empty=_routed_batched(rows[:0],experts,ids[:0],coefficients[:0],shared)
        self.assertEqual(empty.shape,(0,4))

    def test_coefficients_applied_after_completed_expert(self):
        experts=[SimpleNamespace(forward=Mock(side_effect=lambda x:x*3+1)),
                 SimpleNamespace(forward=Mock(side_effect=lambda x:x*5+2))]
        shared=SimpleNamespace(forward=Mock(side_effect=lambda x:x*7))
        x=torch.tensor([[1.,2.],[3.,4.]])
        ids=torch.tensor([[1,1],[1,0]]);weights=torch.tensor([[.25,.5],[0.,.75]])
        actual=_routed(x,experts,ids,weights,shared)
        expected=torch.stack([(x[0]*5+2)*.75+x[0]*7,(x[1]*3+1)*.75+x[1]*7])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for expert in experts:
            for call in expert.forward.call_args_list:self.assertEqual(call.kwargs,{})

    def test_fake_device_and_budget_rejections(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        e=CudaPackedExpert.__new__(CudaPackedExpert)
        e.device=torch.device('cuda:0');e.width=2;e.workspace_bytes=1024;e.activation_limit=None
        e.gate=e.up=e.down=SimpleNamespace(format='e4m3fn_f32')
        with FakeTensorMode():
            dense=CudaPackedGlmFeedForward(e,max_output_bytes=1)
            with self.assertRaises(ValueError):dense(torch.ones(1,1,2,device='cuda:0'))
            with self.assertRaises(ValueError):dense(torch.ones(1,1,2))
            with self.assertRaises(ValueError):CudaPackedGlmFeedForward(e,workspace_bytes=1024)
            with self.assertRaises(ValueError):CudaPackedGlmFeedForward([e,e],shared=e,router=torch.ones(2,2),correction_bias=torch.ones(2))
            with self.assertRaises(ValueError):CudaPackedGlmFeedForward(e,activation_limit=1.)


if __name__=='__main__':unittest.main()
