"""Independent Metal kernels for FP32 Mamba recurrent state."""
import torch

_SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
kernel void conv(device const float* x, device const float* w, device const float* bias,
                 device float* history, device const long* slots, device float* out,
                 constant long& inner, constant long& length, constant long& width, constant long& stride, constant long& reset,
                 uint2 id [[thread_position_in_grid]]) {
    long channel=id.x,row=id.y/length,t=id.y%length;
    if(channel>=inner)return;
    long slot=slots[row];float acc=bias[channel];
    for(long j=0;j<width;++j) {
        float value;
        if(reset) {
            long source=t+j-width+1;
            value=source<0?0:x[(row*length+source)*stride+channel];
        } else {
            value=j<width-1?history[(slot*inner+channel)*width+j+1]:x[row*stride+channel];
            history[(slot*inner+channel)*width+j]=value;
        }
        acc+=value*w[channel*width+j];
    }
    out[(row*length+t)*inner+channel]=acc/(1+exp(-acc));
    if(reset && t==length-1) {
        for(long j=0;j<width;++j) {
            long source=length-width+j;
            history[(slot*inner+channel)*width+j]=source<0?0:x[(row*length+source)*stride+channel];
        }
    }
}
kernel void conv_continue(device const float* x, device const float* w, device const float* bias,
                 device float* history, device const long* slots, device float* out,
                 constant long& inner, constant long& length, constant long& width, constant long& stride,
                 uint2 id [[thread_position_in_grid]]) {
    long channel=id.x,row=id.y;
    if(channel>=inner)return;
    long base=(slots[row]*inner+channel)*width;
    for(long t=0;t<length;++t) {
        float acc=bias[channel];
        for(long j=0;j<width;++j) {
            long source=t+j-width+1;
            float value=source<0?history[base+source+width]:x[(row*length+source)*stride+channel];
            acc+=value*w[channel*width+j];
        }
        out[(row*length+t)*inner+channel]=acc/(1+exp(-acc));
    }
    // Write history after the last output; each channel has one owner.
    for(long j=0;j<width;++j) {
        long source=length-width+j;
        history[base+j]=source<0?history[base+source+width]:x[(row*length+source)*stride+channel];
    }
}
kernel void scan(device const float* x, device const float* gate, device const float* p,
                 device const float* dt, device const float* dt_weight, device const float* bias, device const float* a, device const float* d,
                 device float* state, device const long* slots, device float* out,
                 constant long& inner, constant long& length, constant long& states, constant long& rank,
                 constant long& fused_dt, constant long& gate_stride, constant long& reset, uint2 id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32,row=id.y;
    if(channel>=inner)return;
    long address=(slots[row]*inner+channel)*states+lane;
    float value=reset||lane>=states?0:state[address];
    float decay=lane<states?a[channel*states+lane]:0;
    for(long t=0;t<length;++t) {
        long source=(row*length+t)*inner+channel;
        float step;
        if(fused_dt) {
            float partial=0;
            for(long r=lane;r<rank;r+=32)partial+=dt_weight[channel*rank+r]*p[(row*length+t)*(rank+2*states)+r];
            step=simd_sum(partial);
        } else step=dt[source];
        step+=bias[channel];
        float z=exp(-abs(step)), one_plus=1.0f+z;
        // Metal lacks log1p; compensate the rounding in 1+z for small z.
        step=max(step,0.0f)+(one_plus==1.0f?z:log(one_plus)*z/(one_plus-1.0f));
        float b=lane<states?p[(row*length+t)*(rank+2*states)+rank+lane]:0;
        float c=lane<states?p[(row*length+t)*(rank+2*states)+rank+states+lane]:0;
        value=exp(step*decay)*value+(step*b)*x[source];
        float sum=simd_sum(value*c);
        long gi=(row*length+t)*gate_stride+gate_stride-inner+channel;
        if(lane==0)out[source]=(sum+d[channel]*x[source])*(gate[gi]/(1+exp(-gate[gi])));
    }
    if(lane<states)state[address]=value;
}
'''
_library=None


def library():
    global _library
    if _library is None:_library=torch.mps.compile_shader(_SOURCE)
    return _library


def conv(x,weight,bias,history,slots,reset):
    batch,length,stride=x.shape;inner=history.shape[1]
    out=torch.empty((batch,length,inner),device=x.device,dtype=x.dtype)
    if not reset and length>1:
        library().conv_continue(x,weight,bias,history,slots,out,inner,length,weight.shape[-1],stride,
            threads=(((inner+255)//256)*256,batch),group_size=(256,1))
    else:
        library().conv(x,weight,bias,history,slots,out,inner,length,weight.shape[-1],stride,int(reset),
            threads=(((inner+255)//256)*256,batch*length),group_size=(256,1))
    return out


def scan(x,gate,params,dt,bias,a,d,state,slots,reset,dt_weight=None):
    batch,length,inner=x.shape;s=state.shape[-1];rank=params.shape[-1]-2*s
    out=torch.empty_like(x)
    library().scan(x,gate,params,x if dt is None else dt,x if dt_weight is None else dt_weight,
        bias,a,d,state,slots,out,inner,length,s,rank,int(dt_weight is not None),gate.shape[-1],int(reset),
        threads=(((inner+7)//8)*256,batch),group_size=(256,1))
    return out
