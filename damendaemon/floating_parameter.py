"""Ordinary weight payloads must not imply an undeclared packed format."""


def require_floating(value,name):
    import torch
    if value.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise ValueError('unsupported floating checkpoint parameter dtype: '+name)
    return value
