"""Explicit accelerator selection and memory lifecycle for CUDA, experimental ROCm and Apple MPS."""
import os


def selected_accelerator():
    device = os.environ.get('DLLM_DEVICE', 'cuda')
    if device not in ('cuda', 'rocm', 'mps'):
        raise ValueError('DLLM_DEVICE must be cuda, rocm or mps')
    return device


def selected_device():
    # HIP tensors intentionally use PyTorch's CUDA device namespace.
    accelerator = selected_accelerator()
    return 'cuda' if accelerator == 'rocm' else accelerator


def synchronize():
    import torch
    from .partition_policy import selected_split
    if selected_split() is not None:
        for device in range(2):torch.cuda.synchronize(device)
        return
    (torch.mps if selected_device() == 'mps' else torch.cuda).synchronize()


def empty_cache():
    import torch
    from .partition_policy import selected_split
    if selected_split() is not None:
        for device in range(2):
            with torch.cuda.device(device):torch.cuda.empty_cache()
        return
    (torch.mps if selected_device() == 'mps' else torch.cuda).empty_cache()


def available_memory():
    import torch
    if selected_device() == 'mps':
        return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
    if selected_accelerator() == 'rocm':
        from .rocm_policy import available_memory as rocm_available_memory
        return rocm_available_memory(torch)
    return torch.cuda.mem_get_info()[0]


def attention(q, k, v, **kwargs):
    import torch
    from torch.nn import functional as F
    expand_gqa = q.device.type == 'mps' or (q.device.type == 'cuda'
        and q.dtype == torch.float32 and q.ndim == 4 and q.shape[-2] >= 256)
    if expand_gqa and kwargs.get('enable_gqa', False):
        # Native GQA selects the quadratic-memory math backend for CUDA FP32.
        # Expanding only K/V lets SDPA select its fused efficient kernel; short
        # requests retain their existing path and avoid this materialization.
        # Preserve SDPA's validation for malformed head counts.
        if k.shape[-3] == v.shape[-3] and q.shape[-3] % k.shape[-3] == 0:
            kwargs.pop('enable_gqa')
            repeats = q.shape[-3] // k.shape[-3]
            if repeats != 1:
                k, v = k.repeat_interleave(repeats, dim=-3), v.repeat_interleave(repeats, dim=-3)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def device_name():
    import torch
    return 'Apple MPS' if selected_device() == 'mps' else torch.cuda.get_device_name()
