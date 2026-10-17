import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_deepseek_layer_stage import CudaDeepSeekLayerStage
from tests.test_floating_block import fixture


class CudaDeepSeekStageAdmissionTests(unittest.TestCase):
    def test_cpu_block_rejected(self):
        block,_,_=fixture()
        with self.assertRaises(ValueError):CudaDeepSeekLayerStage([block],start=0,total=1,vocab_size=4)

    def test_budget_before_token_scan(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        stage=object.__new__(CudaDeepSeekLayerStage)
        stage.device=torch.device('cuda:0');stage.streams=2;stage.dim=8
        stage.last=False;stage.max_output_bytes=1;stage.workspace_bytes=1024**2
        with FakeTensorMode(),patch('damendaemon.deepseek_layer_stage._DeepSeekLayerStage._input',side_effect=AssertionError('payload before admission')):
            with self.assertRaisesRegex(ValueError,'budget'):stage._input(torch.empty(1,2,device='cuda',dtype=torch.int64),None,False)
