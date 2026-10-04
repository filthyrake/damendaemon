"""Opt-in 32-value symmetric INT8 blocks with FP16 scales.

The scale and round-away-from-zero convention follows the published Q8_0
representation. This module does not parse GGUF files or delegate inference.
"""
import os
import torch

GROUP = 32


def mode():
    value = os.environ.get('DLLM_QUANTIZATION', 'none')
    if value not in ('none', 'q8_0'):
        raise ValueError('DLLM_QUANTIZATION must be none or q8_0')
    return value


@torch.inference_mode()
def quantize(weight):
    if weight.ndim != 2 or weight.shape[1] % GROUP:
        raise ValueError('Q8_0 requires matrix rows divisible by 32')
    rows, width = weight.shape
    data = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty((rows, width//GROUP), device=weight.device, dtype=torch.float16)
    # Keep conversion workspace bounded even for very large vocabulary matrices.
    for start in range(0, rows, 128):
        block = weight[start:start+128].float().reshape(-1, GROUP)
        if not torch.isfinite(block).all().item():
            raise ValueError('cannot quantize non-finite weights')
        scale = block.abs().amax(-1, keepdim=True) / 127
        if (scale > torch.finfo(torch.float16).max).any().item():
            raise ValueError('Q8_0 scale exceeds FP16 range')
        inverse = torch.where(scale == 0, 0, scale.reciprocal())
        normalized = block * inverse
        rounded = normalized.sign() * torch.floor(normalized.abs() + .5)
        data[start:start+128] = rounded.clamp(-127,127).to(torch.int8).reshape(-1,width)
        scales[start:start+128] = scale.reshape(-1,width//GROUP)
    return data, scales


def dequantize(data, scales, dtype=torch.float32):
    return (data.reshape(*scales.shape, GROUP).float() * scales.float().unsqueeze(-1)).reshape(data.shape).to(dtype)


def prepare(weights, quantization=None):
    scales = {}
    if quantization is None:
        quantization = mode()
    if quantization not in ('none', 'q8_0'):
        raise ValueError('unsupported weight quantization')
    if quantization == 'q8_0':
        for name, weight in list(weights.items()):
            if weight.ndim == 2 and name != 'embed':
                weights[name], scales[name] = quantize(weight)
    return scales


class Linear:
    def __init__(self, weights, host_scales, device, fallback):
        self.scales = {id(weights[name]): value.to(device) for name, value in host_scales.items()}
        self.fallback = fallback

    def __call__(self, x, weight, bias=None):
        if weight.dtype != torch.int8:
            return self.fallback(x, weight, bias)
        from .quantized_kernels import linear
        return linear(x, weight, self.scales[id(weight)], bias)
