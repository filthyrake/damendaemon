"""Incremental CPU tokenizer ByteLevel decoding without whole-history copies.

asm-rewrite branch: fully hand-ported to arm64 assembly
(native/bytelevel_stream_asm.s), including the BYTE_VALUES alphabet;
this file is a delegation shim.
"""
from .bytelevel_stream_asm import BYTE_VALUES, ByteLevelStream

__all__ = ['BYTE_VALUES', 'ByteLevelStream']
