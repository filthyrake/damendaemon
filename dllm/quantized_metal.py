"""Native Metal Q8_0 projection with FP16/FP32 activation and FP32 accumulation."""
import torch
from torch.nn import functional as F

_SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
kernel void expand(device const char* w, device const half* scales, device float* out,
                   constant long& count, uint id [[thread_position_in_grid]]) {
    if(id<count)out[id]=float(w[id])*float(scales[id/32]);
}
kernel void linear(device const float* x, device const char* w, device const half* scales,
                   device const float* bias, device float* out,
                   constant long& rows, constant long& outputs, constant long& width,
                   constant long& biased, uint2 id [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32,batch=id.y*4;
    if(channel>=outputs)return;
    device const char4* weights=reinterpret_cast<device const char4*>(w+channel*width);
    device const float4* input=reinterpret_cast<device const float4*>(x+batch*width);
    float4 sum=0;
    for(long d=lane;d<width/4;d+=32) {
        float4 weight=float4(weights[d])*float(scales[channel*(width/32)+d/8]);
        sum.x+=dot(weight,input[d]);
        if(batch+1<rows)sum.y+=dot(weight,input[width/4+d]);
        if(batch+2<rows)sum.z+=dot(weight,input[2*width/4+d]);
        if(batch+3<rows)sum.w+=dot(weight,input[3*width/4+d]);
    }
    sum=float4(simd_sum(sum.x),simd_sum(sum.y),simd_sum(sum.z),simd_sum(sum.w));
    if(biased)sum+=bias[channel];
    if(lane==0) {
        out[batch*outputs+channel]=sum.x;
        if(batch+1<rows)out[(batch+1)*outputs+channel]=sum.y;
        if(batch+2<rows)out[(batch+2)*outputs+channel]=sum.z;
        if(batch+3<rows)out[(batch+3)*outputs+channel]=sum.w;
    }
}
kernel void expand_half(device const char* w, device const half* scales, device half* out,
                        constant long& count, uint id [[thread_position_in_grid]]) {
    if(id<count)out[id]=half(float(w[id])*float(scales[id/32]));
}
// Match FP16 projection coefficients and outputs while accumulating in FP32.
kernel void linear_half(device const half* x, device const char* w, device const half* scales,
                        device const half* bias, device half* out,
                        constant long& rows, constant long& outputs, constant long& width,
                        constant long& biased, uint2 id [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32,batch=id.y*4;
    if(channel>=outputs)return;
    device const char4* weights=reinterpret_cast<device const char4*>(w+channel*width);
    device const half4* input=reinterpret_cast<device const half4*>(x+batch*width);
    float4 sum=0;
    for(long d=lane;d<width/4;d+=32) {
        float4 weight=float4(half4(float4(weights[d])*float(scales[channel*(width/32)+d/8])));
        sum.x+=dot(weight,float4(input[d]));
        if(batch+1<rows)sum.y+=dot(weight,float4(input[width/4+d]));
        if(batch+2<rows)sum.z+=dot(weight,float4(input[2*width/4+d]));
        if(batch+3<rows)sum.w+=dot(weight,float4(input[3*width/4+d]));
    }
    sum=float4(simd_sum(sum.x),simd_sum(sum.y),simd_sum(sum.z),simd_sum(sum.w));
    if(biased)sum+=float(bias[channel]);
    if(lane==0) {
        out[batch*outputs+channel]=half(sum.x);
        if(batch+1<rows)out[(batch+1)*outputs+channel]=half(sum.y);
        if(batch+2<rows)out[(batch+2)*outputs+channel]=half(sum.z);
        if(batch+3<rows)out[(batch+3)*outputs+channel]=half(sum.w);
    }
}
'''
# Keep the wide-index implementation for layouts beyond the 32-bit element range.
_WIDE_SOURCE = _SOURCE
_INDEX32_SOURCE = (_SOURCE[_SOURCE.index('kernel void linear('):_SOURCE.index('kernel void expand_half(')]
                   + _SOURCE[_SOURCE.index('kernel void linear_half('):])
for _name in ('rows', 'outputs', 'width', 'biased'):
    _INDEX32_SOURCE = _INDEX32_SOURCE.replace('constant long& '+_name, 'constant long& '+_name+'_arg')
_INDEX32_SOURCE = (_INDEX32_SOURCE.replace('void linear(', 'void linear_index32(')
    .replace('void linear_half(', 'void linear_half_index32(')
    .replace('long channel=id.x/32,batch=id.y*4;',
             'uint rows=uint(rows_arg), outputs=uint(outputs_arg), width=uint(width_arg), biased=uint(biased_arg); '
             'uint channel=id.x/32,batch=id.y*4;')
    .replace('for(long d=lane;', 'for(uint d=lane;'))
_SOURCE += _INDEX32_SOURCE
_library=None


def library():
    global _library
    if _library is None:_library=torch.mps.compile_shader(_SOURCE)
    return _library


def linear(x,weight,scales,bias=None):
    if x.dtype not in (torch.float16,torch.float32):raise ValueError('Metal Q8 requires FP16 or FP32 activations')
    half=x.dtype==torch.float16
    if bias is not None:bias=bias.to(x.dtype)
    shape=x.shape[:-1];x=x.reshape(-1,x.shape[-1]).contiguous()
    if x.shape[0]>4:
        expanded=torch.empty(weight.shape,device=x.device,dtype=x.dtype)
        (library().expand_half if half else library().expand)(weight,scales,expanded,weight.numel(),threads=weight.numel())
        out=F.linear(x,expanded,bias)
    else:
        out=torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
        shaders=library()
        narrow=max(weight.numel(),x.numel(),x.shape[0]*weight.shape[0])<2**32
        kernel=(shaders.linear_half_index32 if half else shaders.linear_index32) if narrow else (
            shaders.linear_half if half else shaders.linear)
        kernel(x,weight,scales,x if bias is None else bias,out,
            x.shape[0],weight.shape[0],weight.shape[1],int(bias is not None),
            threads=(((weight.shape[0]+7)//8)*256,(x.shape[0]+3)//4),group_size=(256,1))
    return out.reshape(*shape,weight.shape[0])
