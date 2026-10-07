"""Floating projected compressor composition, before QAT/cache publication."""
import math

import torch
from torch.nn import functional as F

from .compressed_pool import pool_projected
from .deepseek_position import rotate_pairs


@torch.no_grad()
def compress_hidden(hidden, value_weight, gate_weight, offsets, norm_weight, inverse_frequencies,
                    *, state=None, epsilon=1e-6):
    """Return storage-dtype compressed vectors, replacement state and positions.

    Projection/norm parameters are explicitly resident FP32 tensors, matching the
    pinned reference's convenient floating representation. No checkpoint weights
    are implicitly converted here. Input chunks bound projection activations.
    Pooling is FP32; its result casts to input storage dtype BEFORE FP32 RMSNorm,
    casts back, then rotates at block START positions. This preserves the reference
    cast/position order. QAT/Hadamard transformations remain separate and required
    before claiming checkpoint-compatible cache output. Prior state is immutable.
    """
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] < 1
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('compressor hidden state must be floating [batch,sequence,width]')
    if (not isinstance(offsets, torch.Tensor) or offsets.ndim != 2 or offsets.dtype != torch.float32
            or offsets.device != hidden.device or not 1 <= offsets.shape[0] <= 128 or offsets.shape[1] < 1):
        raise ValueError('compressor offsets must be same-device FP32 [ratio,channels]')
    ratio, channels = offsets.shape
    if ratio == 4 and channels % 2:
        raise ValueError('overlapping compression requires paired channel halves')
    width = channels//2 if ratio == 4 else channels
    for tensor, shape in ((value_weight, (channels, hidden.shape[-1])),
                          (gate_weight, (channels, hidden.shape[-1])), (norm_weight, (width,))):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                or tensor.dtype != torch.float32 or tensor.device != hidden.device):
            raise ValueError('compressor parameters must have matching FP32 shape and device')
    if (not isinstance(inverse_frequencies, torch.Tensor) or inverse_frequencies.ndim != 1
            or inverse_frequencies.dtype != torch.float32 or inverse_frequencies.device != hidden.device
            or not 1 <= inverse_frequencies.numel() <= width//2):
        raise ValueError('compressor rotary frequencies must fit an even suffix of the output width')
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon <= torch.finfo(torch.float32).max:
        raise ValueError('compressor normalization epsilon must be finite and positive')
    source = hidden.float()
    pooled, replacement, starts = pool_projected(F.linear(source, value_weight), F.linear(source, gate_weight), offsets, state=state)
    rounded = pooled.to(hidden.dtype).float()
    normalized = (norm_weight*(rounded*(rounded.square().mean(-1, keepdim=True)+epsilon).rsqrt())).to(hidden.dtype)
    rotary_width = inverse_frequencies.numel()*2
    result = normalized.clone()
    result[..., -rotary_width:] = rotate_pairs(normalized[..., -rotary_width:], starts, inverse_frequencies)
    return result, replacement, starts
