"""Direct next-token choices, conditional on an explicit candidate set.

Inspired by typed-decision interfaces (TypeSafe Jev) and indexed output heads
(NInfer); independently implemented. These are not calibrated confidence scores
and do not implement a trained Jev decision head or multi-token answer scoring.
"""
from .scoring_asm import options

__all__ = ['options']
