"""Atomic JSON evidence checkpoints; no hardware or runtime initialization."""
import json
from pathlib import Path
import tempfile


def save_report(path, report):
    """Replace only after a complete write; keep prior evidence on failure.

    Unique same-directory temporaries also permit nested signal checkpoints.
    This provides atomic replacement, not fsync/power-loss durability.
    """
    path = Path(path)
    temporary = None
    try:
        # A signal handler can checkpoint while the main thread is writing.
        # Unique temporaries prevent its report from sharing an open inode.
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

