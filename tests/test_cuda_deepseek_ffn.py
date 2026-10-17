import unittest
import torch
from dllm.cuda_deepseek_ffn import CudaPackedDeepSeekFeedForward


class CudaDeepSeekFfnAdmissionTests(unittest.TestCase):
    def test_unprepared_banks_fail_before_tensor_work(self):
        for experts in (None,[],[None],[None,None]):
            with self.assertRaises(ValueError):
                CudaPackedDeepSeekFeedForward(experts,router=None,shared=None)

    def test_cpu_hidden_rejected_before_execution(self):
        ffn=object.__new__(CudaPackedDeepSeekFeedForward)
        ffn.dim=8;ffn.device=torch.device('cuda:0')
        with self.assertRaises(ValueError):ffn(torch.zeros(1,2,8))

    def test_device_routing_flag_is_bounded_boolean(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from dllm.cuda_packed_expert import CudaPackedExpert
        expert=object.__new__(CudaPackedExpert)
        expert.device=torch.device('cuda:0');expert.width=8
        expert.workspace_bytes=65536;expert.activation_limit=10.
        expert.gate=expert.up=expert.down=SimpleNamespace(format='e4m3fn')
        with FakeTensorMode(), patch('dllm.cuda_deepseek_ffn.finite',side_effect=lambda value,*args,**kwargs:value):
            router=torch.empty(3,8,device='cuda')
            for flag in (True,False):
                ffn=CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=torch.empty(3,device='cuda'),
                    shared=expert,top_k=2,device_routing=flag)
                self.assertIs(ffn.device_routing,flag)
            with self.assertRaises(ValueError):
                CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=torch.empty(3,device='cuda'),
                    shared=expert,top_k=2,device_routing='yes')

    def test_composition_dispatch_reaches_packed_moe(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from dllm.cuda_packed_expert import CudaPackedExpert
        expert=object.__new__(CudaPackedExpert)
        expert.device=torch.device('cuda:0');expert.width=8
        expert.workspace_bytes=65536;expert.activation_limit=10.
        expert.gate=expert.up=expert.down=SimpleNamespace(format='e4m3fn')
        with FakeTensorMode(), patch('dllm.cuda_deepseek_ffn.finite',side_effect=lambda value,*args,**kwargs:value):
            router=torch.empty(3,8,device='cuda');bias=torch.empty(3,device='cuda')
            for composition,kwargs in (('per_token',{'batched':False,'fused':False}),
                                       ('batched',{'batched':True,'fused':False}),
                                       ('fused',{'batched':False,'fused':True})):
                with self.subTest(composition=composition):
                    ffn=CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=bias,shared=expert,
                        top_k=2,composition=composition)
                    self.assertEqual(ffn._composition_kwargs(),kwargs)
            for bad in ('threads','BATCHED',None,2):
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=bias,shared=expert,
                        top_k=2,composition=bad)

    def test_preparation_budget_and_policy_precede_value_scans(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from dllm.cuda_packed_expert import CudaPackedExpert
        expert=object.__new__(CudaPackedExpert)
        expert.device=torch.device('cuda:0');expert.width=8
        expert.workspace_bytes=65536;expert.activation_limit=10.
        expert.gate=expert.up=expert.down=SimpleNamespace(format='e4m3fn')
        with FakeTensorMode(), patch('dllm.cuda_deepseek_ffn.finite',side_effect=AssertionError('scan before admission')):
            router=torch.empty(3,8,device='cuda');bias=torch.empty(3,device='cuda')
            with self.assertRaises(ValueError):
                CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=bias,shared=expert,top_k=2,workspace_bytes=1)
            with self.assertRaises(ValueError):
                CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=bias,shared=expert,top_k=True)
            with self.assertRaises(ValueError):
                CudaPackedDeepSeekFeedForward([expert]*3,router=router,correction_bias=bias,shared=expert,top_k=2,
                    token_table=torch.empty(4,2,dtype=torch.int64,device='cuda'))
