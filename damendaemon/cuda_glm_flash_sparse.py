"""Explicit CUDA Flash NoPE sparse layer with transactional cache publication."""
from .glm_flash_sparse import _FlashSparseAttention
from .cuda_glm_flash_indexer import CudaFlashIndexer


class CudaFlashSparseAttention(_FlashSparseAttention):
    def __init__(self,weights,indexer,**options):
        if not isinstance(indexer,CudaFlashIndexer):
            raise ValueError('CUDA Flash sparse attention requires an explicit CUDA indexer')
        super().__init__(weights,indexer,device=indexer.device,indexer_type=CudaFlashIndexer,**options)
