"""Independent Metal head-wise Mamba-2 selective recurrence."""
import torch

_SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
kernel void scan(device const float* x, device const float* gate, device const float* dt,
                 device const float* bias, device const float* a, device const float* d,
                 device float* state, device const long* slots, device float* out,
                 constant long& inner, constant long& length, constant long& states,
                 constant long& head_dim, constant long& reset,
                 uint2 id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32,row=id.y;
    if(channel>=inner)return;
    long head=channel/head_dim,base=(slots[row]*inner+channel)*states;
    float value[8];
    for(long j=0;j<8;++j) {
        long s=j*32+lane;
        value[j]=reset||s>=states?0:state[base+s];
    }
    for(long t=0;t<length;++t) {
        long source=(row*length+t)*(inner+2*states);
        float step=dt[(row*length+t)*(inner/head_dim)+head]+bias[head];
        float z=exp(-abs(step)),one_plus=1.0f+z;
        step=max(step,0.0f)+(one_plus==1.0f?z:log(one_plus)*z/(one_plus-1.0f));
        float decay=exp(step*a[head]),signal=x[source+channel],sum=0;
        for(long j=0;j<8;++j) {
            long s=j*32+lane;
            if(s<states) {
                value[j]=decay*value[j]+(step*x[source+inner+s])*signal;
                sum+=value[j]*x[source+inner+states+s];
            }
        }
        float result=simd_sum(sum)+d[head]*signal;
        long gi=(row*length+t)*inner+channel;
        if(lane==0)out[gi]=result*(gate[gi]/(1+exp(-gate[gi])));
    }
    for(long j=0;j<8;++j) {
        long s=j*32+lane;
        if(s<states)state[base+s]=value[j];
    }
}
'''
_library=None


def library():
    global _library
    if _library is None:_library=torch.mps.compile_shader(_SOURCE)
    return _library


def scan(x,gate,dt,bias,a,d,state,slots,head_dim,reset):
    batch,length,inner=gate.shape;s=state.shape[-1];out=torch.empty_like(gate)
    library().scan(x,gate,dt,bias,a,d,state,slots,out,inner,length,s,head_dim,int(reset),
        threads=(((inner+7)//8)*256,batch),group_size=(256,1))
    return out
