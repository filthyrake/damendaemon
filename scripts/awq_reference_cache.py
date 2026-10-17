"""Content identities for reusable independent AWQ oracle logits."""
import hashlib
from pathlib import Path
from damendaemon.awq_model import plan


def fingerprint(path):
    path = Path(path).resolve()
    prepared = plan(path)
    hashes = {}
    for file in prepared.identities:
        if file.exists():
            with file.open('rb') as stream:
                hashes[str(file.relative_to(path))] = hashlib.file_digest(stream, 'sha256').hexdigest()
        else:
            hashes[str(file.relative_to(path))] = None
    prepared.check_source()
    return hashes
