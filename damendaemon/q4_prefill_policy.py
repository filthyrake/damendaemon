"""Hardware-bounded Q4 prompt launch policy backed by real serving evidence.

Delegation shim — decision logic lives in native/q4_prefill_policy_asm.s.
"""
from .q4_prefill_policy_asm import _is_a2, use_small_tile  # noqa: F401
