"""Qualified Q4 Marlin layout plan, calculated before GPU weight transfer."""
from dataclasses import dataclass, replace

import torch

from .native_q4_residual import qualified_scope as residual_scope


@dataclass(frozen=True)
class LayoutPlan:
    names: tuple = ()
    canonical_bytes: int = 0
    prepared_bytes: int = 0
    overhead_bytes: int = 0
    repack_peak_bytes: int = 0

    def without_zero_points(self):
        return replace(self, prepared_bytes=self.canonical_bytes, overhead_bytes=0)


def plan(config, weights, scales, quantization, device, dtype, precision, context,
         max_batch, fused, device_name):
    if (not residual_scope(config,quantization,device,dtype,precision,context)
            or context<=0 or not 1<=max_batch<=4 or not fused or device_name!='NVIDIA A2'):
        return LayoutPlan()
    names = ('head',)+tuple(f'{layer}.{part}' for layer in range(32)
                            for part in ('qkv.weight','out','gate_up','down'))
    shapes = {'head':(32000,4096),'qkv.weight':(6144,4096),
              'out':(4096,4096),'gate_up':(28672,4096),'down':(4096,14336)}
    seen_weights,seen_scales = set(),set()
    canonical,overhead,peak = 0,0,0
    for name in names:
        weight,scale = weights.get(name),scales.get(name)
        outputs,width = shapes[name if name=='head' else name.split('.',1)[1]]
        bias_name = name.removesuffix('.weight')+'.bias'
        if (weight is None or scale is None or weight.dtype!=torch.uint8
                or scale.dtype!=torch.float16 or tuple(weight.shape)!=(outputs,width//2)
                or tuple(scale.shape)!=(outputs,width//32) or not weight.is_contiguous()
                or not scale.is_contiguous() or bias_name in weights
                or id(weight) in seen_weights or id(scale) in seen_scales
                or weight is weights.get('embed')):
            return LayoutPlan()
        seen_weights.add(id(weight));seen_scales.add(id(scale))
        weight_bytes = weight.numel()
        scale_bytes = scale.numel()*scale.element_size()
        canonical += weight_bytes+scale_bytes
        overhead += (width//32)*(outputs//8)*4
        # New packed output, transposed and permuted scales, 64 int64 indices.
        # The whole-layout overhead is reserved separately, including zeros.
        peak = max(peak,weight_bytes+2*scale_bytes+64*8)
    return LayoutPlan(names,canonical,canonical+overhead,overhead,peak)
