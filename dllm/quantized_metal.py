"""Native Metal Q8_0 projection with FP32 activation/accumulation."""
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
'''
_library=None


def library():
    global _library
    if _library is None:_library=torch.mps.compile_shader(_SOURCE)
    return _library


def linear(x,weight,scales,bias=None):
    if x.dtype!=torch.float32:raise ValueError('Metal Q8 currently requires FP32 activations')
    shape=x.shape[:-1];x=x.reshape(-1,x.shape[-1]).contiguous()
    if x.shape[0]>4:
        expanded=torch.empty(weight.shape,device=x.device,dtype=torch.float32)
        library().expand(weight,scales,expanded,weight.numel(),threads=weight.numel())
        out=F.linear(x,expanded,bias)
    else:
        out=torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=torch.float32)
        library().linear(x,weight,scales,x if bias is None else bias,out,
            x.shape[0],weight.shape[0],weight.shape[1],int(bias is not None),
            threads=(((weight.shape[0]+7)//8)*256,(x.shape[0]+3)//4),group_size=(256,1))
    return out.reshape(*shape,weight.shape[0])
