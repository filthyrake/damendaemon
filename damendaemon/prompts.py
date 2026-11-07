"""Shared raw-prompt validation and chat/raw dispatch.

asm-rewrite branch: fully hand-ported to arm64 assembly
(native/prompts_asm.s); this file is a delegation shim.
"""
from .prompts_asm import (validate_lengths, decoder_prefix, validate_raw,
                          encode, decode_output)

__all__ = ['validate_lengths', 'decoder_prefix', 'validate_raw', 'encode', 'decode_output']
