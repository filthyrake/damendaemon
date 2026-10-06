"""Validation-only CPU FP32 reference with upstream GGUF-packed Linear storage.

Transformers owns model execution. Each Linear expands independently through
upstream gguf-py for one operation, avoiding full-model FP32 residency. No dllm
quantizers, kernels or model math are used here.
"""
import torch
from torch import nn


class PackedReferenceLinear(nn.Module):
    def __init__(self, original, quantization):
        super().__init__()
        import numpy as np
        from gguf.quants import quantize
        from gguf.constants import GGMLQuantizationType
        self.kind = getattr(GGMLQuantizationType, quantization.upper())
        self.in_features = original.in_features
        self.out_features = original.out_features
        packed = None
        for start in range(0, self.out_features, 128):
            block = original.weight.detach()[start:start + 128].float().cpu().numpy()
            converted = quantize(block, self.kind)
            if packed is None:
                packed = np.empty((self.out_features, converted.shape[1]), dtype=converted.dtype)
            packed[start:start + len(converted)] = converted
        self.register_buffer('packed', torch.from_numpy(packed))
        self.register_buffer('bias', None if original.bias is None else original.bias.detach().float().cpu())

    def forward(self, inputs):
        from gguf.quants import dequantize
        if inputs.device.type != 'cpu' or inputs.dtype != torch.float32:
            raise ValueError('packed reference requires CPU FP32 inputs')
        weights = torch.from_numpy(dequantize(self.packed.numpy(), self.kind))
        return torch.nn.functional.linear(inputs, weights, self.bias)


def replace_linears(model, quantization):
    # Traverse a name snapshot; retaining module objects would retain all source
    # matrices until the whole model had been converted.
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name in names:
        if name.startswith(('model.visual.', 'mtp.')):
            continue
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, child)
        setattr(parent, child, PackedReferenceLinear(original, quantization))
        del original
    return model.float()
