"""Reuse one serialized warmup stream per device across model residency changes.

cuBLAS retains workspaces by stream. Creating a new warmup stream for every
executor can retain additional workspaces after the model itself is unloaded.
"""
from contextlib import contextmanager
import threading

import torch

_states = {}
_states_lock = threading.Lock()


@contextmanager
def warmup_stream():
    device = torch.cuda.current_device()
    with _states_lock:
        if device not in _states:
            _states[device] = (torch.cuda.Stream(device=device), threading.RLock())
        stream, lock = _states[device]
    with lock:
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            current.wait_stream(stream)
