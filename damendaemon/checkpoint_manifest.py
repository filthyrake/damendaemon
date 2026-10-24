"""Checkpoint manifest digest: cross-rank checkpoint agreement for mixed chains.

The gate-4 staging discipline ships a crc32.txt next to every staged
checkpoint copy. Hashing that manifest gives every rank a 32-byte value
derived from its OWN copy's file: identical checkpoint copies derive the
identical digest (regardless of the per-rank weight format each rank
loads), so a wrong or stale copy aborts startup. A checkpoint staged
without a manifest yields None — the chain contract then cannot detect a
wrong checkpoint, and callers must verify copies out of band.
"""
import hashlib
from pathlib import Path

MANIFEST_NAME = 'crc32.txt'


def manifest_digest(checkpoint):
    """32-byte SHA-256 of the staged checkpoint's CRC manifest, or None."""
    manifest = Path(checkpoint) / MANIFEST_NAME
    if not manifest.is_file():
        return None
    return hashlib.sha256(MANIFEST_NAME.encode() + b'\0'
                          + manifest.read_bytes()).digest()
