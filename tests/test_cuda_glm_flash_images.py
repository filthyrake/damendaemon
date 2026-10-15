import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import torch
from dllm.cuda_glm_flash_image_placement import place_cuda_flash_image
from dllm.cuda_glm_flash_image_generation import CudaFlashImageGenerator,cuda_image_embeddings
from tests.test_glm_flash_image_generation import vision,text_fixture,ScriptedImageModel
from tests.test_cuda_glm_flash_placement import fixture


class CudaFlashImagesTests(unittest.TestCase):
    def test_joint_budget_and_vision_metadata_precede_any_copy(self):
        model=fixture();encoder=vision(model.embedding.shape[1])
        with patch('dllm.cuda_glm_flash_image_placement._copy',side_effect=AssertionError('copy before admission')):
            with self.assertRaises(ValueError):place_cuda_flash_image(model,encoder,device='cuda:0',max_bytes=1)
            encoder.w['downsample.bias']=torch.zeros(1)
            with self.assertRaises(ValueError):place_cuda_flash_image(model,encoder,device='cuda:0',max_bytes=10**8)

    def test_cpu_model_cannot_enter_cuda_image_path(self):
        with tempfile.TemporaryDirectory() as temp:
            text=text_fixture(Path(temp));model=ScriptedImageModel(text,'abc')
            with self.assertRaises(ValueError):CudaFlashImageGenerator(model,text,vision(1))
            with self.assertRaises(ValueError):cuda_image_embeddings(model,torch.ones(1,1,dtype=torch.int64),[torch.zeros(0,1)],image_token_id=text.image_token_id)
            self.assertFalse(hasattr(CudaFlashImageGenerator,'generate_batch_images'))

    def test_vision_budget_precedes_patch_transfer(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from dllm.cuda_glm_flash_vision import CudaFlashVision
        prepared=vision(4)
        runner=object.__new__(CudaFlashImageGenerator);runner.device=torch.device('cuda:0')
        with FakeTensorMode():
            weights={k:torch.empty(v.shape,device='cuda') for k,v in prepared.w.items()}
            runner.vision=CudaFlashVision(prepared.c,weights,max_workspace_bytes=1)
            patches=torch.empty(4,1176)
            with patch.object(torch.Tensor,'to',side_effect=AssertionError('transfer before admission')):
                with self.assertRaisesRegex(ValueError,'budget'):runner._image_features(patches,[(1,2,2)])
