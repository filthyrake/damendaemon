"""ASM-backed checkpoint_manifest shim; logic lives in native/checkpoint_manifest_asm.s."""
from .checkpoint_manifest_asm import MANIFEST_NAME, _manifest_digest as manifest_digest

__all__ = ['MANIFEST_NAME', 'manifest_digest']
