"""Canonical model locations without losing split-checkpoint sibling names.

asm-rewrite branch: fully hand-ported to arm64 assembly
(native/model_path_asm.s); this file is a delegation shim.
"""
from .model_path_asm import canonical_model_path

__all__ = ['canonical_model_path']
