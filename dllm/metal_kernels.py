"""Native Metal kernels with FP32 arithmetic and optional FP16 weight storage.

Requires torch.mps.compile_shader. No CPU fallback or Triton dependency.
"""
import torch
from torch.nn import functional as F

if not hasattr(torch.mps, 'compile_shader'):
    raise ImportError('native Metal kernels require a PyTorch build with compile_shader')

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void rms(device const float* x, device const float* w, device float* y,
                constant long& rows, constant long& width, constant float& eps,
                uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= rows) return;
    float ss = 0;
    for (uint d = lane; d < width; d += 32) ss += x[row*width+d]*x[row*width+d];
    float inv = rsqrt(simd_sum(ss)/float(width)+eps);
    for (uint d = lane; d < width; d += 32) y[row*width+d] = x[row*width+d]*inv*w[d];
}

kernel void swiglu(device const float* x, device float* y, constant long& count,
                   constant long& width, uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint offset = (id/width)*2*width + id%width;
    float gate = x[offset];
    y[id] = gate/(1.0f+exp(-gate))*x[offset+width];
}

template <typename Weight>
inline void linear_impl(device const float* x, device const Weight* w, device const float* bias,
                        device float* y, long m, long n, long k, long has_bias,
                        uint2 id, uint lane) {
    uint row = id.x/32, batch = id.y*4;
    if (row >= n) return;
    float4 sums = 0;
    if (k%4 == 0) {
        device const vec<Weight,4>* weights = reinterpret_cast<device const vec<Weight,4>*>(w+row*k);
        device const float4* inputs = reinterpret_cast<device const float4*>(x+batch*k);
        for (uint d = lane; d < k/4; d += 32) {
            float4 weight = float4(weights[d]);
            sums.x += dot(weight,inputs[d]);
            if (batch+1 < m) sums.y += dot(weight,inputs[k/4+d]);
            if (batch+2 < m) sums.z += dot(weight,inputs[2*k/4+d]);
            if (batch+3 < m) sums.w += dot(weight,inputs[3*k/4+d]);
        }
    } else {
        for (uint d = lane; d < k; d += 32) {
            float weight = w[row*k+d];
            sums.x += weight*x[batch*k+d];
            if (batch+1 < m) sums.y += weight*x[(batch+1)*k+d];
            if (batch+2 < m) sums.z += weight*x[(batch+2)*k+d];
            if (batch+3 < m) sums.w += weight*x[(batch+3)*k+d];
        }
    }
    sums = float4(simd_sum(sums.x),simd_sum(sums.y),simd_sum(sums.z),simd_sum(sums.w));
    if (has_bias) sums += bias[row];
    if (lane == 0) {
        y[batch*n+row] = sums.x;
        if (batch+1 < m) y[(batch+1)*n+row] = sums.y;
        if (batch+2 < m) y[(batch+2)*n+row] = sums.z;
        if (batch+3 < m) y[(batch+3)*n+row] = sums.w;
    }
}

kernel void linear(device const float* x, device const float* w, device const float* bias,
                   device float* y, constant long& m, constant long& n, constant long& k,
                   constant long& has_bias, uint2 id [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    linear_impl(x,w,bias,y,m,n,k,has_bias,id,lane);
}

kernel void linear_half(device const float* x, device const half* w, device const float* bias,
                        device float* y, constant long& m, constant long& n, constant long& k,
                        constant long& has_bias, uint2 id [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    linear_impl(x,w,bias,y,m,n,k,has_bias,id,lane);
}

kernel void rope(device const float* x, device const float* cosine, device const float* sine,
                 device const long* positions, device const long* slots, device float* q,
                 device float* cache_k, device float* cache_v,
                 device const float* qw, device const float* kw,
                 constant long& heads, constant long& kvheads, constant long& dim,
                 constant long& context, constant long& normalize, constant long& single,
                 constant float& eps, uint2 id [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    uint head = id.x/32, token = id.y;
    if (head >= heads+2*kvheads) return;
    long pos = positions[token], slot = single ? 0 : slots[token];
    long base = (token*(heads+2*kvheads)+head)*dim;
    bool query = head < heads, key = head >= heads && head < heads+kvheads;
    if (!query && !key) {
        long offset = ((slot*kvheads+head-heads-kvheads)*context+pos)*dim;
        for (uint d=lane; d<dim; d+=32) cache_v[offset+d] = x[base+d];
        return;
    }
    float inv=1;
    if (normalize) {
        float ss=0;
        for (uint d=lane; d<dim; d+=32) ss+=x[base+d]*x[base+d];
        inv=rsqrt(simd_sum(ss)/float(dim)+eps);
    }
    for (uint d=lane; d<dim; d+=32) {
        uint other = (d+dim/2)%dim;
        float a=x[base+d]*inv, b=x[base+other]*inv;
        if (normalize) { a*=query?qw[d]:kw[d]; b*=query?qw[other]:kw[other]; }
        float value=a*cosine[pos*dim+d]+(d<dim/2?-b:b)*sine[pos*dim+d];
        if (query) q[(token*heads+head)*dim+d]=value;
        else cache_k[((slot*kvheads+head-heads)*context+pos)*dim+d]=value;
    }
}

kernel void attention(device const float* q, device const float* k, device const float* v,
                      device const long* positions, device const long* slots, device float* out,
                      constant long& heads, constant long& kvheads, constant long& dim,
                      constant long& context, constant long& single,
                      uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scores[4096];
    threadgroup float partial[8];
    uint head=group.x, row=group.y, lane=tid%32, simd=tid/32;
    long end=positions[row], slot=single?0:slots[row];
    long base=(slot*kvheads+head/(heads/kvheads))*context*dim;
    long qb=(row*heads+head)*dim;
    for (uint p=simd; p<=end; p+=8) {
        float dot=0;
        for (uint d=lane; d<dim; d+=32) dot+=q[qb+d]*k[base+p*dim+d];
        dot=simd_sum(dot);
        if(lane==0) scores[p]=dot*rsqrt(float(dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;
    for(uint p=tid; p<=end; p+=256) maximum=max(maximum,scores[p]);
    maximum=simd_max(maximum);
    if(lane==0) partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=-INFINITY;
    for(uint j=0;j<8;j++) maximum=max(maximum,partial[j]);
    // All readers of partial must finish before its next reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;
    for(uint p=tid;p<=end;p+=256) { float e=exp(scores[p]-maximum); scores[p]=e; total+=e; }
    total=simd_sum(total);
    if(lane==0) partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=0;
    for(uint j=0;j<8;j++) total+=partial[j];
    for(uint d=tid;d<dim;d+=256) {
        float sum=0;
        for(uint p=0;p<=end;p++) sum+=scores[p]*v[base+p*dim+d];
        out[qb+d]=sum/total;
    }
}
'''
_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def rms(x, weight, eps, precise=False):
    x = x.contiguous()
    out = torch.empty_like(x)
    rows, width = x.numel()//x.shape[-1], x.shape[-1]
    library().rms(x, weight, out, rows, width, eps, threads=((rows+7)//8*256), group_size=256)
    return out


def swiglu(gate_up, precise=False):
    out = torch.empty((gate_up.shape[0], gate_up.shape[-1]//2), device='mps', dtype=torch.float32)
    library().swiglu(gate_up, out, out.numel(), out.shape[-1], threads=out.numel())
    return out


def linear(x, weight, bias=None):
    if x.shape[0] > 4:
        return F.linear(x, weight.float(), bias)
    out = torch.empty((x.shape[0], weight.shape[0]), device='mps', dtype=torch.float32)
    kernel = library().linear_half if weight.dtype == torch.float16 else library().linear
    kernel(x, weight, bias if bias is not None else x, out,
        x.shape[0], weight.shape[0], weight.shape[1], int(bias is not None),
        threads=(((weight.shape[0]+7)//8)*256, (x.shape[0]+3)//4), group_size=(256,1))
    return out


def rope_batch(qkv, cos, sin, positions, slots, cache_k, cache_v, heads,
               qw=None, kw=None, eps=1.e-6, single=False):
    n, dim = qkv.shape[0], cache_k.shape[-1]
    kvheads, context = cache_k.shape[1:3]
    q = torch.empty((n, heads, 1, dim), device='mps', dtype=torch.float32)
    library().rope(qkv, cos, sin, positions, slots, q, cache_k, cache_v,
        qw if qw is not None else cos, kw if kw is not None else cos,
        heads, kvheads, dim, context, int(qw is not None), int(single), eps,
        threads=(((heads+2*kvheads+7)//8)*256,n), group_size=(256,1))
    return q


def rope_store(qkv, cos, sin, positions, cache_k, cache_v, heads, qw=None, kw=None, eps=1.e-6):
    return rope_batch(qkv, cos, sin, positions, positions, cache_k, cache_v,
                      heads, qw, kw, eps, single=True).permute(2,1,0,3)


def attention(q, cache_k, cache_v, positions, slots=None):
    n, heads, _, dim = q.shape
    kvheads, context = cache_k.shape[1:3]
    if context > 4096:
        from .device import attention as tensor_attention
        actual_slots = slots if slots is not None else torch.zeros_like(positions)
        mask = torch.arange(context, device='mps')[None,None,None,:] <= positions[:,None,None,None]
        return tensor_attention(q, cache_k.index_select(0,actual_slots), cache_v.index_select(0,actual_slots),
                                attn_mask=mask, enable_gqa=True)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    library().attention(q.contiguous(), cache_k, cache_v, positions,
        slots if slots is not None else positions, out, heads, kvheads, dim, context, int(slots is None),
        threads=(heads*256,n), group_size=(256,1))
    return out
