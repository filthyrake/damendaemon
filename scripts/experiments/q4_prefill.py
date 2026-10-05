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
