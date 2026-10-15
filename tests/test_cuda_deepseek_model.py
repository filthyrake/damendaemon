import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from dllm.cuda_deepseek_model import CudaDeepSeekModel


class CudaDeepSeekModelAdmissionTests(unittest.TestCase):
    def test_cpu_model_rejected(self):
        with self.assertRaises(ValueError):CudaDeepSeekModel(torch.zeros(4,8),[],{})

    def test_output_budget_precedes_token_scan(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        model=object.__new__(CudaDeepSeekModel)
        model.device=torch.device('cuda:0');model.workspace_bytes=1024**2;model.max_output_bytes=1
        model.blocks=[SimpleNamespace(hc_mult=2)];model.head={}
        with FakeTensorMode():
            model.embedding=torch.empty(4,8,device=model.device)
            with self.assertRaisesRegex(ValueError,'budget'):
                model._admit(torch.empty(1,2,dtype=torch.int64,device=model.device))

    def test_aggregate_cache_rejection_precedes_embedding(self):
        model=object.__new__(CudaDeepSeekModel)
        model.blocks=[SimpleNamespace(cache_bytes_per_slot=lambda capacity:100)]
        with patch.object(model,'_admit'),patch.object(model,'_embed',side_effect=AssertionError('embedding before admission')):
            with self.assertRaisesRegex(ValueError,'cache'):
                model.prefill(torch.ones(1,2,dtype=torch.int64),max_tokens=3,max_cache_bytes=99)
