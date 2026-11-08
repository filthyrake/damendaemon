"""Explicit CUDA Flash NoPE sparse layer with transactional cache publication.

Delegation shim — decision logic lives in native/cuda_glm_flash_sparse_asm.s.
"""
from .cuda_glm_flash_sparse_asm import (  # noqa: F401
    CudaFlashSparseAttention, CudaFlashIndexer, _FlashSparseAttention)
