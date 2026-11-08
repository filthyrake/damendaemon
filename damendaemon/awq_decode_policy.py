"""A2 small-row AWQ selection bounded by measured projection shapes.

Delegation shim — decision logic lives in native/awq_decode_policy_asm.s.
"""
from .awq_decode_policy_asm import _ROWS, _is_a2, use_small_rows  # noqa: F401
