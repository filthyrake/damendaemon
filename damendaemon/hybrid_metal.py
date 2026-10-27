"""Independent Metal gated-delta scan for the qualified Qwen3.5 hybrid path."""
import torch

_SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
kernel void scan(device const float* q, device const float* k,
                 device const float* v, device const float* decay,
                 device const float* beta, device float* state,
                 device const long* slots, device float* out,
                 constant long& length, constant long& heads,
                 constant long& values, constant long& reset,
                 uint2 id [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    long channel=id.x/32, row=id.y/heads, head=id.y%heads;
    if(channel>=values)return;
    long base=(slots[row]*heads+head)*128*values+channel;
    float s[4];
    for(uint j=0;j<4;++j)s[j]=reset?0:state[base+(lane+32*j)*values];
    for(long t=0;t<length;++t) {
        long entry=(row*length+t)*heads+head;
        float factor=exp(decay[entry]);
        float keys[4],queries[4],dot=0;
        for(uint j=0;j<4;++j) {
            keys[j]=k[entry*128+lane+32*j];
            queries[j]=q[entry*128+lane+32*j];
            s[j]*=factor;
            dot+=s[j]*keys[j];
        }
        float delta=(v[entry*values+channel]-simd_sum(dot))*beta[entry];
        float result=0;
        for(uint j=0;j<4;++j) {
            s[j]+=keys[j]*delta;
            result+=s[j]*queries[j];
        }
        result=simd_sum(result);
        if(lane==0)out[entry*values+channel]=result;
    }
    for(uint j=0;j<4;++j)state[base+(lane+32*j)*values]=s[j];
}
'''
_library=None


def library():
    global _library
    if _library is None:_library=torch.mps.compile_shader(_SOURCE)
    return _library


def eligible(config,context,max_batch=1):
    # The qualified geometry set: the kernel is written against 16 linear
    # heads of 128 key / 128 value channels, which the 0.8B and 2B Qwen3.5
    # text backbones share exactly (hidden_size differs, but the scan never
    # touches it). Each entry carries its own qualified context bound —
    # the scan length checks cover up to that bound.
    geometries = {
        1024: 1024,  # Qwen3.5-0.8B text: unchanged — its qualified Apple
                     # evidence (text+images, context 1024) is the bound;
                     # >1024 on the 0.8B needs its own replay cell.
        2048: 32768,  # Qwen3.5-2B text (#4: scan shapes shared with the 0.8B,
                      # verified against the CPU oracle at lengths to 32K and
                      # by chunked whole-model golden replays at 4096 and
                      # 32768 on the pinned revision).
    }
    expected = dict(model_type='qwen3_5_text', num_hidden_layers=24,
                    linear_key_head_dim=128, linear_value_head_dim=128,
                    linear_num_key_heads=16, linear_num_value_heads=16)
    if not all(config.get(key) == value for key, value in expected.items()):
        return False
    bound = geometries.get(config.get('hidden_size'))
    return (bound is not None and 1 <= context <= bound and 1 <= max_batch <= 4)


def scan(q,k,v,decay,beta,state,slots,reset):
    """Accept normalized/scaled FP32 q/k, updating selected FP32 state slots."""
    batch,length,heads,keys=q.shape
    values=v.shape[-1]
    if keys!=128 or q.device.type!='mps':raise ValueError('candidate requires MPS with 128 key channels')
    out=torch.empty((batch,length,heads,values),device=q.device,dtype=torch.float32)
    library().scan(q.contiguous(),k.contiguous(),v.float().contiguous(),decay.contiguous(),
        beta.contiguous(),state,slots,out,length,heads,values,int(reset),
        threads=(values*32,batch*heads),group_size=(32,1))
    return out
