"""ASM-backed Harmony shim; logic lives in native/harmony_asm.s."""
from .harmony_asm import TextStream, output_stream, _MARKERS

__all__ = ['TextStream', 'output_stream', '_MARKERS']
