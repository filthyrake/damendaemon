"""Atomic publication of trusted, locally generated reference traces."""
import os
from pathlib import Path
import tempfile

import torch


def save_trace(path, metadata, events, *, complete):
    destination = Path(path)
    if not complete:
        destination = destination.with_name(destination.stem + '.partial' + destination.suffix)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+'.',
                                         suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            torch.save({**metadata, 'trace': events, 'complete': complete}, output)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
