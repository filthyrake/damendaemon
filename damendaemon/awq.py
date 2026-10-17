"""Packed AWQ GEMM coefficients; no checkpoint or runtime activation yet.

AWQ stores eight output channels per int32 in the order decoded by shifts
[0, 4, 1, 5, 2, 6, 3, 7]. Zero points use the same packing, without GPTQ's
historical plus-one convention. This module retains the checkpoint layout.
"""
from dataclasses import dataclass

import torch


def group_size(config):
    allowed = {'bits', 'group_size', 'quant_method', 'version', 'zero_point',
               'modules_to_not_convert'}
    if (not isinstance(config, dict) or set(config) - allowed
            or config.get('quant_method') != 'awq' or type(config.get('bits')) is not int or config['bits'] != 4
            or type(config.get('group_size')) is not int or config['group_size'] != 128
            or not isinstance(config.get('version'), str) or config['version'].lower() != 'gemm'
            or config.get('zero_point') is not True
            or config.get('modules_to_not_convert') not in (None, [])):
        raise ValueError('requires AWQ GEMM, four bits, group size 128 and zero points, without module exclusions')
    return config['group_size']


def unpack_channels(packed):
    """Coefficient inspection helper; inference must keep the packed layout."""
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError('AWQ channels require a rank-two int32 tensor')
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device) * 4
    return ((packed[..., None] >> shifts) & 15).reshape(packed.shape[0], -1)


@dataclass(frozen=True)
class Matrix:
    weight: torch.Tensor
    zeros: torch.Tensor
    scales: torch.Tensor
    inputs: int
    outputs: int
    group: int = 128

    def __post_init__(self):
        if (type(self.inputs) is not int or type(self.outputs) is not int
                or self.inputs <= 0 or self.outputs <= 0 or type(self.group) is not int or self.group != 128
                or self.inputs % self.group or self.outputs % 8):
            raise ValueError('unsupported AWQ matrix geometry')
        expected = ((self.inputs, self.outputs // 8),
                    (self.inputs // self.group, self.outputs // 8),
                    (self.inputs // self.group, self.outputs))
        for tensor, shape, dtype in zip((self.weight, self.zeros, self.scales), expected,
                                        (torch.int32, torch.int32, torch.float16)):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError('invalid AWQ buffer shape, dtype or layout')
            if tensor.device != self.weight.device:
                raise ValueError('AWQ buffers must share a device')
        if not bool(torch.isfinite(self.scales).all()) or bool((self.scales < 0).any()):
            raise ValueError('AWQ scales must be finite and nonnegative')

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.weight, self.zeros, self.scales))

    def dequantize(self, dtype=torch.float16):
        """Explicit validation helper, not an inference/storage fallback."""
        values = unpack_channels(self.weight).float()
        zeros = unpack_channels(self.zeros).repeat_interleave(self.group, dim=0)
        scales = self.scales.float().repeat_interleave(self.group, dim=0)
        return ((values - zeros) * scales).to(dtype).T.contiguous()


def fuse(matrices):
    """Fuse output channels while retaining their original packed bits."""
    matrices = tuple(matrices)
    if not matrices or any(m.inputs != matrices[0].inputs or m.group != matrices[0].group
                           or m.weight.device != matrices[0].weight.device for m in matrices):
        raise ValueError('AWQ fusion requires compatible input groups and devices')
    return Matrix(*(torch.cat([getattr(m, name) for m in matrices], dim=1)
                    for name in ('weight', 'zeros', 'scales')),
                  matrices[0].inputs, sum(m.outputs for m in matrices), matrices[0].group)
