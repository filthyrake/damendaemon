"""A2 small-row AWQ selection bounded by measured projection shapes."""
from functools import lru_cache
import torch

_ROWS = {(896, 1152): 2, (896, 896): 4, (896, 9728): 3, (4864, 896): 4}


@lru_cache(maxsize=None)
def _is_a2(device):
    return torch.cuda.get_device_properties(device).name == 'NVIDIA A2'


def use_small_rows(rows, inputs, outputs, device):
    if not 1 <= rows <= _ROWS.get((inputs, outputs), 0):
        return False
    return device.type == 'cuda' and _is_a2(device)
