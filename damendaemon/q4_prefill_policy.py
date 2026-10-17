"""Hardware-bounded Q4 prompt launch policy backed by real serving evidence."""
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _is_a2(device):
    return torch.cuda.get_device_properties(device).name == 'NVIDIA A2'


def use_small_tile(rows, outputs, width, device):
    # Keep decode and unqualified projections free of device-property lookups.
    if not 32 < rows <= 64 or (width, outputs) not in (
            (4096, 6144), (4096, 4096), (4096, 28672),
            (14336, 4096), (4096, 32000)):
        return False
    return device.type == 'cuda' and _is_a2(device)
