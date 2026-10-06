"""Validation-only FP32 reference with upstream GGUF-packed Linear storage.

Transformers owns model execution. CPU expansion uses upstream gguf-py; GPU
expansion uses independently checked Torch operations, one matrix at a time.
Neither path uses dllm quantizers, kernels or model math.
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


def expand_packed_torch(packed, kind):
    """Expand GGUF Q8_0/Q4_0 with ordinary Torch operations on the input device.

    Independent validation code: storage comes from upstream gguf-py. Each
    block has a little-endian FP16 scale followed by 32 signed Q8 values, or
    16 bytes containing the low-half/high-half Q4 values biased by eight.
    Tests compare these operations exactly with upstream CPU dequantization.
    """
    from gguf.constants import GGMLQuantizationType
    sizes = {GGMLQuantizationType.Q8_0: 34, GGMLQuantizationType.Q4_0: 18}
    if kind not in sizes:
        raise ValueError('Torch reference expansion supports only Q8_0/Q4_0')
    size = sizes[kind]
    if packed.dtype != torch.uint8 or packed.ndim != 2 or not packed.is_contiguous():
        raise ValueError('packed reference storage must be a contiguous uint8 matrix')
    if packed.shape[1] % size:
        raise ValueError('packed reference rows must contain whole blocks')
    blocks = packed.reshape(-1, size)
    scales = blocks[:, :2].contiguous().view(torch.float16).float()
    payload = blocks[:, 2:]
    if kind == GGMLQuantizationType.Q8_0:
        values = payload.view(torch.int8).float()
    else:
        values = torch.cat((payload & 15, payload >> 4), dim=1).float() - 8
    return (values * scales).reshape(packed.shape[0], packed.shape[1] // size * 32)


class PackedGpuReferenceLinear(PackedReferenceLinear):
    """Independent Torch GPU expansion and FP32 linear, one matrix at a time."""
    def forward(self, inputs):
        if inputs.device.type != 'cuda' or inputs.dtype != torch.float32:
            raise ValueError('packed GPU reference requires CUDA/ROCm FP32 inputs')
        weights = expand_packed_torch(self.packed, self.kind)
        return torch.nn.functional.linear(inputs, weights, self.bias)


def replace_linears(model, quantization, *, gpu_arithmetic=False):
    # Traverse a name snapshot; retaining module objects would retain all source
    # matrices until the whole model had been converted.
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name in names:
        if name.startswith(('model.visual.', 'mtp.')):
            continue
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, child)
        linear = PackedGpuReferenceLinear if gpu_arithmetic else PackedReferenceLinear
        setattr(parent, child, linear(original, quantization))
        del original
    return model.float()
