"""Explicit CUDA FP32 Flash linear attention with transactional owned state.

Delegation shim — decision logic lives in native/cuda_glm_flash_linear_asm.s.
"""
from .cuda_glm_flash_linear_asm import (  # noqa: F401
    CudaFlashLinearAttention, _FlashLinearAttention, _device)
