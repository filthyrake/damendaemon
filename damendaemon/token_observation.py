"""Constant-memory host token observations, not isolated engine/kernel timing.

asm-rewrite branch: observe and report are fully hand-ported to arm64
assembly (native/token_observation_asm.s), including the float math;
this file is the declarative dataclass plus delegation shims.
"""
from dataclasses import dataclass

from .token_observation_asm import observe as _observe, report as _report


@dataclass
class TokenObservation:
    count: int = 0
    first: float | None = None
    last: float | None = None
    max_gap: float = 0.0
    sequential: bool = True

    def observe(self, offset, *, sequential=True):
        # Offsets come from the owning runtime's monotonic clock. Speculative
        # bursts have no trustworthy per-token completion timestamps.
        return _observe(self, offset, sequential=sequential)

    def report(self):
        return _report(self)
