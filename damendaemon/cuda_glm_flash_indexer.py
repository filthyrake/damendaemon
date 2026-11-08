"""Explicit CUDA FP32 Flash learned pool selection.

Delegation shim — decision logic lives in native/cuda_glm_flash_indexer_asm.s.
"""
from .cuda_glm_flash_indexer_asm import CudaFlashIndexer, _FlashIndexer, _device  # noqa: F401
