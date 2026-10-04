"""Explicit accelerator selection and memory lifecycle for CUDA and Apple MPS."""
import os


def selected_device():
    device = os.environ.get('DLLM_DEVICE', 'cuda')
    if device not in ('cuda', 'mps'):
        raise ValueError('DLLM_DEVICE must be cuda or mps')
    return device


def synchronize():
    import torch
    (torch.mps if selected_device() == 'mps' else torch.cuda).synchronize()


def empty_cache():
    import torch
    (torch.mps if selected_device() == 'mps' else torch.cuda).empty_cache()


def available_memory():
    import torch
    if selected_device() == 'mps':
        return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
    return torch.cuda.mem_get_info()[0]


def attention(q, k, v, **kwargs):
    from torch.nn import functional as F
    if q.device.type == 'mps' and kwargs.pop('enable_gqa', False):
        repeats = q.shape[-3] // k.shape[-3]
        k, v = k.repeat_interleave(repeats, dim=-3), v.repeat_interleave(repeats, dim=-3)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def device_name():
    import torch
    return 'Apple MPS' if selected_device() == 'mps' else torch.cuda.get_device_name()
