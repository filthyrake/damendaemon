"""Explicit experimental single-APU admission; no NVIDIA native libraries."""
import math
import os
from pathlib import Path

NVIDIA_LIBRARIES = (
    'DLLM_Q4_CUDA_LIBRARY', 'DLLM_KQUANT_CUDA_LIBRARY',
    'DLLM_Q4_RESIDUAL_LIBRARY', 'DLLM_KQUANT_RESIDUAL_LIBRARY',
    'DLLM_KQUANT_RESIDUAL_F32_LIBRARY', 'DLLM_Q4_MARLIN_LIBRARY',
    'DLLM_AWQ_MARLIN_LIBRARY',
)


def configure_attention():
    """Enable the qualified gfx1151 SDPA path early in an owned process.

    Runtime validation below restricts ROCm execution to gfx1151. Keep an
    explicit zero as a diagnostic fallback; do not edit the host environment.
    """
    key = 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'
    if os.environ.get(key, '1') not in ('0', '1'):
        raise ValueError(f'{key} must be 0 or 1')
    os.environ.setdefault(key, '1')


def validate_runtime(torch):
    if not getattr(torch.version, 'hip', None):
        raise RuntimeError('ROCm selection requires a HIP build of PyTorch')
    physical = os.environ.get('ROCR_VISIBLE_DEVICES', '')
    if (not physical.isascii() or not physical.isdecimal()
            or os.environ.get('HIP_VISIBLE_DEVICES') != '0'
            or os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('select one AMD ordinal with ROCR_VISIBLE_DEVICES, set HIP_VISIBLE_DEVICES=0, and unset CUDA_VISIBLE_DEVICES')
    if os.environ.get('DLLM_EXPERIMENTAL_LAYER_SPLIT') is not None:
        raise ValueError('ROCm multi-GPU execution is not qualified')
    if any(os.environ.get(name) for name in NVIDIA_LIBRARIES):
        raise ValueError('NVIDIA native CUDA libraries cannot be selected for ROCm')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one HIP GPU must be visible')
    architecture = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '').split(':')[0]
    if architecture != 'gfx1151':
        raise ValueError('experimental ROCm admission currently targets gfx1151')
    memory_limit()


def memory_limit():
    value = float(os.environ.get('DLLM_ROCM_MEMORY_GIB', '8'))
    if not math.isfinite(value) or value <= 0 or not math.isfinite(value * 1024**3):
        raise ValueError('DLLM_ROCM_MEMORY_GIB must be finite and positive')
    return int(value * 1024**3)


def host_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        words = line.split()
        if words and words[0] == 'MemAvailable:' and len(words) == 3 and words[2] == 'kB':
            return int(words[1]) * 1024
    raise RuntimeError('ROCm APU admission requires Linux MemAvailable accounting')


HOST_RESERVE = 16 * 1024**3


def require_host_headroom():
    available = host_available()
    if available <= HOST_RESERVE:
        raise MemoryError('ROCm execution would exhaust the 16 GiB host reserve')
    return available - HOST_RESERVE


def configure_allocator(torch):
    """Cap the caching allocator; an admission estimate alone is not a limit.

    HIP/driver allocations outside this allocator still require independent
    host monitoring. Never loosen an already stricter caller-supplied cap.
    """
    headroom = require_host_headroom()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    previous = torch.cuda.get_per_process_memory_fraction(0)
    if total <= 0 or not math.isfinite(previous) or not 0 < previous <= 1:
        raise RuntimeError('cannot establish ROCm allocator budget')
    budget = min(memory_limit(), total * previous, free + reserved, headroom + reserved)
    if budget <= 0 or reserved > budget:
        raise MemoryError('existing ROCm allocations exceed the configured budget')
    torch.cuda.set_per_process_memory_fraction(float(budget / total), 0)
    return int(budget)


def available_memory(torch):
    validate_runtime(torch)
    free, total = torch.cuda.mem_get_info()
    # APU free-memory reporting may exceed the reported device total. Bound by
    # both, the explicit process budget, and live host headroom. Allocator-owned
    # library workspaces count against the process budget, even after unload.
    return max(0, min(free, total,
                      memory_limit() - torch.cuda.memory_allocated(),
                      host_available() - 16 * 1024**3))


def prefer_q8_small_batch_matvec(torch, device, rows):
    """Qualified gfx1151 dispatch; other architectures retain their policy.

    Used while launching/capturing a Q8 projection, not during graph replay.
    The caller checks FP16 input precision before selecting this policy.
    """
    return (2 <= rows <= 4 and bool(torch.version.hip)
            and torch.cuda.get_device_properties(device).gcnArchName.split(':', 1)[0] == 'gfx1151')
