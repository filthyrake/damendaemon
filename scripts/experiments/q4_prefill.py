"""Validation-only contiguous-activation, single-dot Q4 prefill tile."""
import triton as tr
import triton.language as tl
from dllm.quantized_kernels_q4 import _values


@tr.jit
def single_dot(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    offsets=tl.arange(0,BK)
    accum=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        columns=start*BK+offsets
        x=tl.load(X+rows[:,None]*K+columns[None,:],(rows[:,None]<M)&(columns[None,:]<K),other=0)
        w=_values(W,S,outputs[:,None],columns[None,:],N,K).to(tl.float16)
        accum+=tl.dot(x,tl.trans(w))
    if BIASED:accum+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],accum,(rows[:,None]<M)&(outputs[None,:]<N))


@tr.jit
def single_dot_paired_load(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr,
                          ADJACENT:tl.constexpr=False):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    packed_columns=tl.arange(0,BK//2)
    groups=tl.arange(0,BK//32)
    accum=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+columns
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        p=start*(BK//2)+packed_columns
        packed=tl.load(W+outputs[:,None]*(K//2)+p[None,:],
                       (outputs[:,None]<N)&(p[None,:]<K//2),other=0).to(tl.int32)
        g=start*(BK//32)+groups
        scale=tl.load(S+outputs[:,None]*(K//32)+g[None,:],
                      (outputs[:,None]<N)&(g[None,:]<K//32),other=0).to(tl.float32)
        low=(((packed&15)-8).to(tl.float32).reshape(BN,BK//32,16)*scale[:,:,None]).to(tl.float16)
        high=(((packed>>4)-8).to(tl.float32).reshape(BN,BK//32,16)*scale[:,:,None]).to(tl.float16)
        if ADJACENT:w=tl.join(low,high).reshape(BN,BK)
        else:w=tl.join(low,high).permute(0,1,3,2).reshape(BN,BK)
        accum+=tl.dot(x,tl.trans(w))
    if BIASED:accum+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],accum,(rows[:,None]<M)&(outputs[None,:]<N))


@tr.jit
def single_dot_adjacent(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    single_dot_paired_load(X,W,S,BIAS,Y,M,N,K,BM,BN,BK,BIASED,True)


def repack_adjacent(weight):
    """Diagnostic byte permutation; values and scales are unchanged."""
    import torch
    blocks=weight.reshape(weight.shape[0],-1,16)
    values=torch.cat((blocks&15,blocks>>4),dim=-1)
    return (values[...,::2]|(values[...,1::2]<<4)).reshape(weight.shape).contiguous()
