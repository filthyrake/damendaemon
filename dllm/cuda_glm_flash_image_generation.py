"""Explicit CUDA Flash still-image generation; CPU RGB preprocessing stays bounded."""
import torch
from .glm_flash_image_generation import FlashImageReferenceGenerator
from .glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from .glm_flash_image_tool_text import FlashImageToolText
from .glm_flash_image_embeddings import _image_embeddings
from .cuda_glm_flash_model import CudaFlashModel
from .cuda_glm_flash_vision import CudaFlashVision


def cuda_image_embeddings(model,tokens,features,*,image_token_id):
    return _image_embeddings(model,tokens,features,image_token_id=image_token_id,model_types=(CudaFlashModel,))


class CudaFlashImageGenerator(FlashImageReferenceGenerator):
    model_type=CudaFlashModel
    device_type='cuda'
    vision_type=CudaFlashVision
    embed_images=staticmethod(cuda_image_embeddings)

    def _image_features(self,patches,grids):
        self.vision._input_metadata(patches,grids,device=torch.device('cpu'))
        return self.vision.forward(patches.to(self.device),grids)


class CudaFlashImageToolGenerator(CudaFlashImageGenerator):
    text_type=FlashImageToolText
    generate_image_tools=FlashImageToolReferenceGenerator.generate_image_tools
