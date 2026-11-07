"""Measured short-prefill dispatch for native Metal quantized projections."""
from .metal_projection_policy_asm import use_packed_projection

__all__ = ['use_packed_projection']
