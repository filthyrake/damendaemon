"""Explicit CUDA FP32 Flash learned pool selection."""
from .glm_flash_indexer import _FlashIndexer
from .cuda_glm_flash_primitives import _device


class CudaFlashIndexer(_FlashIndexer):
    def __init__(self,weights,**options):
        device=_device(weights.get('wk.weight') if isinstance(weights,dict) else None)
        super().__init__(weights,device=device,**options)
