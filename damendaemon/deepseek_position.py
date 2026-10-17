"""Independent adjacent-pair RoPE and DeepSeek YaRN frequencies.

Equations follow DeepSeek-V4-Flash inference/model.py (MIT), revision
60d8d70770c6776ff598c94bb586a859a38244f1. Unlike some other YaRN schedules,
this contract does not multiply cosine/sine by an attention magnitude factor.
"""
import math

import torch


def frequencies(dim, *, base, original_length=0, factor=1., beta_fast=32., beta_slow=1., device='cpu'):
    if type(dim) is not int or not 2 <= dim <= 4096 or dim % 2:
        raise ValueError('rotary width must be even and in 2..4096')
    if type(original_length) is not int or not 0 <= original_length < 2**31:
        raise ValueError('original context length must be a nonnegative integer below 2**31')
    for value in (base, factor, beta_fast, beta_slow):
        if type(value) not in (int, float) or not 0 < value <= torch.finfo(torch.float32).max:
            raise ValueError('rotary schedule parameters must be finite and positive')
    if base <= 1 or factor < 1 or beta_fast < beta_slow:
        raise ValueError('rotary schedule requires base>1, factor>=1 and beta_fast>=beta_slow')
    inverse = 1./(base**(torch.arange(0, dim, 2, device=device, dtype=torch.float32)/dim))
    if original_length:
        def boundary(rotations):
            return dim*math.log(original_length/(rotations*2*math.pi))/(2*math.log(base))
        low = max(math.floor(boundary(beta_fast)), 0)
        high = min(math.ceil(boundary(beta_slow)), dim-1)
        if low == high:
            high += .001
        blend = ((torch.arange(dim//2, device=device, dtype=torch.float32)-low)/(high-low)).clamp(0, 1)
        inverse = inverse/factor*blend+inverse*(1-blend)
    return inverse


def rotate_pairs(values, positions, inverse_frequencies, *, inverse=False):
    """Rotate adjacent pairs without mutating input; positions may vary by slot."""
    if (not isinstance(values, torch.Tensor) or values.ndim not in (3, 4)
            or values.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or values.shape[-1] < 2 or values.shape[-1] % 2):
        raise ValueError('rotary input must be floating [batch,sequence,(heads),even_width]')
    if (not isinstance(inverse_frequencies, torch.Tensor)
            or inverse_frequencies.shape != (values.shape[-1]//2,)
            or inverse_frequencies.dtype != torch.float32 or inverse_frequencies.device != values.device):
        raise ValueError('rotary frequencies must be matching same-device FP32 vector')
    if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64
            or positions.device != values.device or positions.shape not in (values.shape[:2], (values.shape[1],))
            or type(inverse) is not bool):
        raise ValueError('rotary positions must be same-device int64 sequence or batch/sequence positions')
    angles = positions.float().unsqueeze(-1)*inverse_frequencies
    if positions.ndim == 1:
        angles = angles.unsqueeze(0)
    if values.ndim == 4:
        angles = angles.unsqueeze(-2)
    cosine, sine = angles.cos(), angles.sin()
    if inverse:
        sine = -sine
    pairs = values.float().unflatten(-1, (-1, 2))
    real, imaginary = pairs[..., 0], pairs[..., 1]
    return torch.stack((real*cosine-imaginary*sine, imaginary*cosine+real*sine), -1).flatten(-2).to(values.dtype)
