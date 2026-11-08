"""ASM-backed evidence_environment shim; logic lives in native/evidence_environment_asm.s."""
from .evidence_environment_asm import REPORT_KEYS, _report_environment as report_environment

__all__ = ['REPORT_KEYS', 'report_environment']
