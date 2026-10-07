"""Floating QAT component simulations; no packed projection or GPU qualification.

Scale floors/block geometry follow DeepSeek-V4-Flash inference/kernel.py (MIT),
revision 60d8d70770c6776ff598c94bb586a859a38244f1. E2M1 uses nearest/even rounding;
E4M3FN uses the tensor dtype conversion. Inputs are required to be finite.
"""
import torch


def _input(values):
    if (not isinstance(values, torch.Tensor) or values.ndim < 1 or values.shape[-1] < 1
            or values.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('activation simulation requires floating tensors with nonempty width')


def _ceil_power_two(values):
    # Avoid log2 rounding around exact powers of two.
    mantissa, exponent = torch.frexp(values)
    return torch.ldexp(torch.ones_like(values), exponent-(mantissa == .5).to(exponent.dtype))


def simulate_fp8(values, *, block_size=64):
    """Power-of-two scaled E4M3FN quantize/dequantize, preserving storage dtype."""
    _input(values)
    if type(block_size) is not int or block_size not in (64, 128) or values.shape[-1] % block_size:
        raise ValueError('FP8 simulation needs complete blocks of 64 or 128 values')
    groups = values.float().unflatten(-1, (-1, block_size))
    scale = _ceil_power_two(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)*(1/448))
    quantized = (groups/scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (quantized*scale).flatten(-2).to(values.dtype)


def simulate_fp4(values):
    """E2M1 nearest/even block simulation with one E8M0 scale per 32 values."""
    _input(values)
    if values.shape[-1] % 32:
        raise ValueError('FP4 simulation needs complete blocks of 32 values')
    groups = values.float().unflatten(-1, (-1, 32))
    scale = _ceil_power_two(groups.abs().amax(-1, keepdim=True).clamp_min(6*2**-126)*(1/6))
    normalized = (groups/scale).clamp(-6, 6)
    magnitude = normalized.abs()
    boundaries = values.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32)
    levels = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    code = torch.bucketize(magnitude.contiguous(), boundaries)
    tie = magnitude == boundaries[code.clamp_max(6)]
    code = code+((code % 2 == 1) & tie).long()
    quantized = torch.copysign(levels[code], normalized)
    return (quantized*scale).flatten(-2).to(values.dtype)


def normalized_hadamard(values):
    """Normalized Sylvester transform, with FP32 butterflies and final dtype cast."""
    _input(values)
    width = values.shape[-1]
    if width & (width-1) or width > 4096:
        raise ValueError('Hadamard width must be a power of two at most 4096')
    transformed = values.float()
    stride = 1
    while stride < width:
        pairs = transformed.reshape(*values.shape[:-1], width//(2*stride), 2, stride)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        transformed = torch.stack((left+right, left-right), -2).reshape(values.shape)
        stride *= 2
    return (transformed*(width**-.5)).to(values.dtype)
