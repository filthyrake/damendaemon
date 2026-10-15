import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from dllm.cuda_glm_model import CudaGlmBlock,CudaGlmModel
from dllm.glm_model import GlmModel
from tests.test_glm_packed_model import model_pair


class CudaGlmModelTests(unittest.TestCase):
    def test_cpu_components_cannot_enter_cuda_model(self):
        model,_=model_pair();block=model.blocks[0]
        with self.assertRaises(ValueError):CudaGlmBlock(block.attention,block.feed_forward,block.input_norm,block.post_norm)
        with self.assertRaises(ValueError):CudaGlmModel(model.embedding,model.blocks,model.norm,model.head)

    def test_fake_cuda_prepared_model_device_and_stack_admission(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode(),patch('torch.isfinite',return_value=SimpleNamespace(all=lambda:True)):
            block=CudaGlmBlock.__new__(CudaGlmBlock);block.device=torch.device('cuda:0');block.dim=6
            block.attention=SimpleNamespace(indexer=object(),index_top_k=3)
            embedding=torch.empty(13,6,device='cuda:0');norm=torch.empty(6,device='cuda:0')
            model=CudaGlmModel(embedding,[block],norm,embedding)
            self.assertEqual(model.device,torch.device('cuda:0'))
            with self.assertRaises(ValueError):model._tokens(torch.zeros(1,1,dtype=torch.int64),False)
            with self.assertRaises(ValueError):GlmModel(embedding,[block],norm,embedding)
            with self.assertRaises(ValueError):CudaGlmModel(embedding,[block],norm.cpu(),embedding)
            block.attention.indexer=None
            with self.assertRaises(ValueError):CudaGlmModel(embedding,[block],norm,embedding)


if __name__=='__main__':unittest.main()
