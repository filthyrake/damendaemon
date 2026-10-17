"""Conservative Linux host admission for experimental weight offload.

These observations are not reservations: other processes can allocate between
checks. External memory limits and monitoring remain necessary for qualification.
"""
import math
from pathlib import Path

from safetensors import safe_open

HOST_RESERVE = 16 * 1024**3


def available_memory():
    try:
        entries = [line.split() for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:')]
        if len(entries) != 1:
            raise ValueError('missing or duplicate MemAvailable')
        words = entries[0]
        if (len(words) != 3 or words[2] != 'kB'
                or not words[1].isascii() or not words[1].isdecimal()):
            raise ValueError('invalid MemAvailable')
        return int(words[1]) * 1024
    except (OSError, ValueError) as error:
        raise RuntimeError('hybrid offload requires valid Linux MemAvailable accounting') from error


def require_headroom(pending_bytes=0):
    if type(pending_bytes) is not int or pending_bytes < 0:
        raise ValueError('pending host bytes must be a nonnegative integer')
    available = available_memory()
    if available <= HOST_RESERVE + pending_bytes:
        raise MemoryError('hybrid offload load would exhaust the 16 GiB host reserve')
    return available - HOST_RESERVE


def admit_checkpoint(files):
    """Check source pages plus conservative conversion/cat copies using headers.

    Each source element allows two additional FP32 elements for destination and
    conversion/concatenation temporaries. This deliberately overestimates FP16
    storage and tied weights; it is not an exact peak or allocator guarantee.
    """
    source = converted = 0
    for file in files:
        with safe_open(str(file), framework='pt', device='cpu') as reader:
            for name in reader.keys():
                if name.startswith(('model.visual.', 'mtp.')):
                    continue
                header = reader.get_slice(name)
                dtype = header.get_dtype()
                if dtype not in ('F16', 'BF16', 'F32'):
                    raise ValueError('hybrid offload requires floating-point checkpoint weights')
                count = math.prod(header.get_shape())
                source += count * (4 if dtype == 'F32' else 2)
                converted += count * 8
    require_headroom(source + converted)
    return source + converted
