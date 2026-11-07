"""Inspect equivalent tokenizer decoder sequences without recursive traversal.

asm-rewrite branch: fully hand-ported to arm64 assembly
(native/tokenizer_decoders_asm.s); this file is a delegation shim.
"""
from .tokenizer_decoders_asm import decoder_steps, is_bytelevel

__all__ = ['decoder_steps', 'is_bytelevel']
