"""ASM-backed moe_prefill shim; logic lives in native/moe_prefill_asm.s."""
from .moe_prefill_asm import (_qualified_scope as qualified_scope,
                              _use_grouped_prefill as use_grouped_prefill,
                              _use_packed_prefill as use_packed_prefill)

__all__ = ['qualified_scope', 'use_grouped_prefill', 'use_packed_prefill']
