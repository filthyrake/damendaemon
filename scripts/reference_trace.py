"""Atomic publication of trusted, locally generated reference traces."""
import os
from pathlib import Path
import tempfile

import torch


def validate_replay(trace, quantization, *, allow_quantization_mismatch=False):
    """Reject incomplete/empty or accidentally mismatched qualification inputs."""
    if not trace.get('complete', True):
        raise ValueError('incomplete reference trace; use the completed output')
    events = trace.get('trace', [])
    if not events or any(not event.get('targets') for event in events):
        raise ValueError('reference trace must contain targets for every event')
    reference = trace.get('reference_quantization', 'none')
    if reference != quantization and not allow_quantization_mismatch:
        raise ValueError(f'reference quantization {reference} differs from runtime {quantization}; '
                         'use --allow-quantization-mismatch only for an explicit quality comparison')


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
