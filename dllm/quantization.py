"""Opt-in Q8_0 and packed Q4_0 blocks with FP16 scales.

The conversion conventions follow the published Q8_0 and Q4_0
representations. This module does not parse GGUF files or delegate inference.
"""
import os
import torch

GROUP = 32


def mode(value=None):
    from .management import quantization_mode
    return quantization_mode(value)


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


@torch.inference_mode()
def quantize_q4(weight):
    """Q4_0: signed scale from the first absolute maximum; paired nibbles."""
    if weight.ndim!=2 or weight.shape[1]%GROUP:
        raise ValueError('Q4_0 requires matrix rows divisible by 32')
    rows,width=weight.shape
    data=torch.empty((rows,width//2),device=weight.device,dtype=torch.uint8)
    scales=torch.empty((rows,width//GROUP),device=weight.device,dtype=torch.float16)
    for start in range(0,rows,128):
        block=weight[start:start+128].float().reshape(-1,GROUP)
        if not torch.isfinite(block).all().item():raise ValueError('cannot quantize non-finite weights')
        indices=block.abs().argmax(-1,keepdim=True)
        scale=block.gather(-1,indices)/-8
        if (scale.abs()>torch.finfo(torch.float16).max).any().item():raise ValueError('Q4_0 scale exceeds FP16 range')
        inverse=torch.where(scale==0,0,scale.reciprocal())
        values=torch.trunc(block*inverse+8.5).clamp(0,15).to(torch.uint8)
        packed=values[:,:16] | (values[:,16:]<<4)
        data[start:start+128]=packed.reshape(-1,width//2)
        scales[start:start+128]=scale.reshape(-1,width//GROUP)
    return data,scales


def dequantize_q4(data,scales,dtype=torch.float32):
    blocks=data.reshape(-1,16)
    values=torch.cat((blocks & 15,blocks>>4),dim=-1).float()-8
    return (values*scales.float().reshape(-1,1)).reshape(data.shape[0],data.shape[1]*2).to(dtype)


def prepare(weights, quantization=None):
    scales = {}
    if quantization is None:
        quantization = mode()
    if quantization not in ('none', 'q8_0','q4_0'):
        raise ValueError('unsupported weight quantization')
    if quantization in ('q8_0','q4_0'):
        pack=quantize if quantization=='q8_0' else quantize_q4
        for name, weight in list(weights.items()):
            if weight.ndim == 2 and name != 'embed':
                weights[name], scales[name] = pack(weight)
    return scales


class Linear:
    def __init__(self, weights, host_scales, device, fallback, packed_formats=None, packed_decode=None, packed_prefill=None, q4_decode=None, packed_cache=None, awq_matrices=None):
        copies = {key: value.to(device) for key, value in {id(t):t for t in host_scales.values()}.items()}
        self.scales = {id(weights[name]): copies[id(value)] for name, value in host_scales.items()}
        self.awq = {}
        if awq_matrices:
            from .awq import Matrix
            self.awq = {id(weights[name]): Matrix(weights[name], matrix.zeros.to(device),
                                                 self.scales[id(weights[name])], matrix.inputs, matrix.outputs, matrix.group)
                        for name, matrix in awq_matrices.items()}
        self.formats = {id(weights[name]): layout for name, layout in (packed_formats or {}).items()}
        self.fallback = fallback
        self.packed_decode = packed_decode
        self.packed_prefill = packed_prefill
        self.q4_decode = q4_decode
        self.packed_cache = packed_cache or {}

    def __call__(self, x, weight, bias=None):
        if self.awq and id(weight) in self.awq:
            from .awq_kernels import linear
            return linear(x, self.awq[id(weight)], bias)
        if id(weight) in self.formats:
            cached = self.packed_cache.get(id(weight))
            if cached is not None and x.dtype == torch.float16 and x.numel() // x.shape[-1] > 1:
                return torch.nn.functional.linear(x, cached, bias)
            if self.packed_decode is not None:
                output = self.packed_decode(x, weight, self.formats[id(weight)], bias)
                if output is not None:
                    return output
            if self.packed_prefill is not None:
                output = self.packed_prefill(x, weight, self.formats[id(weight)], bias)
                if output is not None:
                    return output
            from .kquant_kernels import linear
            return linear(x, weight, self.formats[id(weight)], bias)
        if weight.dtype == torch.uint8:
            if self.q4_decode is not None:
                output = self.q4_decode(x, weight, self.scales[id(weight)], bias)
                if output is not None:return output
            from .quantized_kernels_q4 import linear
            return linear(x,weight,self.scales[id(weight)],bias)
        if weight.dtype != torch.int8:
            return self.fallback(x, weight, bias)
        if x.device.type=='mps':
            from .quantized_metal import linear
            return linear(x,weight,self.scales[id(weight)],bias)
        from .quantized_kernels import linear
        return linear(x, weight, self.scales[id(weight)], bias)

    def embedding(self, ids, weight, dtype):
        if id(weight) in self.formats:
            from .kquant_kernels import embedding
            return embedding(ids, weight, self.formats[id(weight)], dtype)
        from .quantized_embedding import embedding
        return embedding(ids, weight, self.scales[id(weight)], dtype)
