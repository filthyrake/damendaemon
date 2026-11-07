"""Text stopping with safe incremental output across token boundaries.

asm-rewrite branch: this module is fully hand-ported to arm64
assembly (native/stopping_asm.s); this file is now a thin delegation
shim for import compatibility. Port state: PORTING_STATUS.md.
"""
from .stopping_asm import stop_sequences, text_delta

__all__ = ['stop_sequences', 'text_delta']
