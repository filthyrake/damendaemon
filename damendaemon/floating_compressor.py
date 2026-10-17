"""Floating projected compressor composition with explicit cache simulations."""
import math
from functools import partial

import torch
from torch.nn import functional as F

from .compressed_pool import pool_projected
from .deepseek_position import rotate_pairs


@torch.no_grad()
def _compress_hidden(hidden, value_weight, gate_weight, offsets, norm_weight, inverse_frequencies,
                    *, state=None, epsilon=1e-6, cache_transform=None,pool_fn=None,validate_finite=False):
    """Return storage-dtype compressed vectors, replacement state and positions.

    Projection/norm parameters are explicitly resident FP32 tensors, matching the
    pinned reference's convenient floating representation. No checkpoint weights
    are implicitly converted here. Input chunks bound projection activations.
    Pooling is FP32; its result casts to input storage dtype BEFORE FP32 RMSNorm,
    casts back, then rotates at block START positions. This preserves the reference
    cast/position order. Select kv_fp8 or index_fp4 explicitly for cache simulation;
    None retains the unquantized reference output. Prior state is immutable.
    Accelerator rounding and checkpoint-level behavior remain unqualified.
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
    rotary_width = inverse_frequencies.numel()*2
    if cache_transform not in (None, 'kv_fp8', 'index_fp4'):
        raise ValueError('compressor cache transform must be kv_fp8, index_fp4 or None')
    if cache_transform == 'kv_fp8' and (width-rotary_width) % 64:
        raise ValueError('KV FP8 simulation requires complete non-rotary blocks of 64')
    if cache_transform == 'index_fp4' and (width % 32 or width & (width-1) or width > 4096):
        raise ValueError('indexer transform requires a power-of-two width in 32..4096')
    def checked(value):
        if validate_finite and not torch.isfinite(value).all():raise ValueError('nonfinite compressor intermediate')
        return value
    if validate_finite:
        for value in (hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies):checked(value)
    source = hidden.float()
    pooling=partial(pool_projected,validate_scores=validate_finite) if pool_fn is None else pool_fn
    pooled, replacement, starts = pooling(checked(F.linear(source, value_weight)),checked(F.linear(source, gate_weight)),offsets,state=state)
    checked(pooled)
    rounded = pooled.to(hidden.dtype).float()
    variance=checked(rounded.square().mean(-1,keepdim=True)+epsilon)
    normalized=checked((norm_weight*(rounded*variance.rsqrt())).to(hidden.dtype))
    rotary_width = inverse_frequencies.numel()*2
    result = normalized.clone()
    result[..., -rotary_width:] = rotate_pairs(normalized[..., -rotary_width:], starts, inverse_frequencies)
    checked(result)
    if cache_transform == 'kv_fp8' and rotary_width < width:
        from .activation_simulation import simulate_fp8
        result[..., :-rotary_width] = simulate_fp8(result[..., :-rotary_width])
    elif cache_transform == 'index_fp4':
        from .activation_simulation import normalized_hadamard, simulate_fp4
        result = simulate_fp4(normalized_hadamard(result))
    return checked(result), replacement, starts


def compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,*,state=None,epsilon=1e-6,cache_transform=None):
    return _compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,
                            state=state,epsilon=epsilon,cache_transform=cache_transform)


def checked_compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,
                            *,state=None,epsilon=1e-6,cache_transform=None):
    """Reference-model boundary: validate arithmetic before cache publication."""
    return _compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,
                            state=state,epsilon=epsilon,cache_transform=cache_transform,validate_finite=True)
