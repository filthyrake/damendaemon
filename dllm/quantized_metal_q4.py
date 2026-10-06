"""Native Metal Q4_0 projections; runtime admission is qualified separately."""
import torch
from torch.nn import functional as F


_KERNELS = r'''
kernel void expand_NAME(device const uchar* w, device const half* scales,
                        device TYPE* out, constant long& count,
                        uint id [[thread_position_in_grid]]) {
    if(id >= count)return;
    long group=id/32, within=id%32;
    uchar packed=w[group*16+within%16];
    int value=int(within<16 ? packed & 15 : packed >> 4)-8;
    out[id]=TYPE(float(value)*float(scales[group]));
}
kernel void linear_NAME(device const TYPE* x, device const uchar* w,
                        device const half* scales, device const TYPE* bias,
                        device TYPE* out, constant long& rows,
                        constant long& outputs, constant long& width,
                        constant long& biased, uint2 id [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32, batch=id.y*4;
    if(channel>=outputs)return;
    device const uchar4* weights=reinterpret_cast<device const uchar4*>(w+channel*(width/2));
    device const TYPE4* input=reinterpret_cast<device const TYPE4*>(x+batch*width);
    float4 sum=0;
    for(long d=lane;d<width/8;d+=32) {
        uchar4 packed=weights[d];
        float scale=float(scales[channel*(width/32)+d/4]);
        float4 low_weight=float4(TYPE4(float4(int4(packed & uchar4(15))-int4(8))*scale));
        float4 high_weight=float4(TYPE4(float4(int4(packed >> 4)-int4(8))*scale));
        long low=(d/4)*8+d%4, high=low+4;
        sum.x+=dot(low_weight,float4(input[low]))+dot(high_weight,float4(input[high]));
        if(batch+1<rows)sum.y+=dot(low_weight,float4(input[width/4+low]))+dot(high_weight,float4(input[width/4+high]));
        if(batch+2<rows)sum.z+=dot(low_weight,float4(input[2*width/4+low]))+dot(high_weight,float4(input[2*width/4+high]));
        if(batch+3<rows)sum.w+=dot(low_weight,float4(input[3*width/4+low]))+dot(high_weight,float4(input[3*width/4+high]));
    }
    sum=float4(simd_sum(sum.x),simd_sum(sum.y),simd_sum(sum.z),simd_sum(sum.w));
    if(biased)sum+=float(bias[channel]);
    if(lane==0) {
        out[batch*outputs+channel]=TYPE(sum.x);
        if(batch+1<rows)out[(batch+1)*outputs+channel]=TYPE(sum.y);
        if(batch+2<rows)out[(batch+2)*outputs+channel]=TYPE(sum.z);
        if(batch+3<rows)out[(batch+3)*outputs+channel]=TYPE(sum.w);
    }
}
'''
_SOURCE = '#include <metal_stdlib>\nusing namespace metal;\n' + ''.join(
    _KERNELS.replace('NAME', name).replace('TYPE', name) for name in ('float', 'half'))
_library = None


def library():
    global _library
    if _library is None:_library = torch.mps.compile_shader(_SOURCE)
    return _library


def linear(x, weight, scales, bias=None):
    if x.device.type != 'mps' or x.dtype not in (torch.float16, torch.float32):
        raise ValueError('Metal Q4 requires MPS FP16 or FP32 activations')
    if (weight.ndim != 2 or weight.dtype != torch.uint8 or weight.device != x.device
            or not weight.is_contiguous() or weight.shape[0] < 1 or weight.shape[1] < 16
            or weight.shape[1] % 16):
        raise ValueError('Metal Q4 requires contiguous packed groups of 32 coefficients')
    outputs, width = weight.shape[0], weight.shape[1]*2
    if (x.ndim < 1 or x.shape[-1] != width or x.numel() == 0
            or scales.shape != (outputs, width//32) or scales.dtype != torch.float16
            or scales.device != x.device or not scales.is_contiguous()):
        raise ValueError('Metal Q4 activation or scale layout mismatch')
    if bias is not None:
        if bias.shape != (outputs,) or bias.device != x.device:raise ValueError('Metal Q4 bias layout mismatch')
        bias = bias.to(x.dtype).contiguous()
    shape = x.shape[:-1]
    x = x.reshape(-1, width).contiguous()
    half = x.dtype == torch.float16
    if x.shape[0] > 4:
        # Expand only this projection temporarily for the MPS matrix product.
        expanded = torch.empty((outputs, width), device=x.device, dtype=x.dtype)
        (library().expand_half if half else library().expand_float)(
            weight, scales, expanded, expanded.numel(), threads=expanded.numel())
        out = F.linear(x, expanded, bias)
    else:
        out = torch.empty((x.shape[0], outputs), device=x.device, dtype=x.dtype)
        (library().linear_half if half else library().linear_float)(
            x, weight, scales, x if bias is None else bias, out,
            x.shape[0], outputs, width, int(bias is not None),
            threads=(((outputs+7)//8)*256, (x.shape[0]+3)//4), group_size=(256, 1))
    return out.reshape(*shape, outputs)
