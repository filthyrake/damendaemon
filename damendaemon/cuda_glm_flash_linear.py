"""Explicit CUDA FP32 Flash linear attention with transactional owned state."""
from .glm_flash_linear import _FlashLinearAttention
from .cuda_glm_flash_primitives import _device


class CudaFlashLinearAttention(_FlashLinearAttention):
    def __init__(self,weights,**options):
        device=_device(weights.get('q_proj.weight') if isinstance(weights,dict) else None)
        super().__init__(weights,device=device,**options)
