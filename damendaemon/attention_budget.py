"""Bound the FP32 partial/LSE storage and the temporary chunk output per launch."""
from .attention_budget_asm import query_rows

__all__ = ['query_rows']
