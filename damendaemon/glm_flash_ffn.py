"""Explicit Flash clipped FFN policies over shared GLM routing equations.

Delegation shim — decision logic lives in native/glm_flash_ffn_asm.s.
"""
from .glm_flash_ffn_asm import FlashFeedForward, PackedFlashFeedForward, _limit  # noqa: F401
