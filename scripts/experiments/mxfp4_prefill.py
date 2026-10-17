"""Experiment: expand packed expert weights inside a multi-row Tensor Core tile."""
import torch
import triton as tr
import triton.language as tl
from damendaemon.precise_matmul import _split
from damendaemon.mxfp4_kernels import _value_normal


@tr.jit
def _project(HI,LO,S,W,WS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK);acc=tl.full((BM,BN),0,tl.float32)
    scale=tl.load(S+rows,rows<M,other=1.)
    for start in range(tl.cdiv(K,BK)):
        indices=start*BK+k
        mask=(rows[:,None]<M)&(indices[None,:]<K)
        hi=tl.load(HI+rows[:,None]*K+indices[None,:],mask,other=0)
        lo=tl.load(LO+rows[:,None]*K+indices[None,:],mask,other=0)
        mask=(cols[None,:]<N)&(indices[:,None]<K)
        packed=tl.load(W+cols[None,:]*(K//2)+indices[:,None]//2,mask,other=0).to(tl.int32)
        exponent=tl.load(WS+cols[None,:]*(K//32)+indices[:,None]//32,mask,other=127).to(tl.int32)
        code=(packed>>((indices[:,None]&1)*4))&15
        weight=_value_normal(code,exponent).to(tl.float16)
        partial=tl.dot(hi,weight)+tl.dot(lo,weight)
        acc+=partial*scale[:,None]
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def linear(x,blocks,scales):
    """Caller must validate immutable scales are in the exact FP16 range 104..140."""
    rows,width=x.shape
    hi=torch.empty_like(x,dtype=torch.float16);lo=torch.empty_like(hi)
    factor=torch.empty(rows,device=x.device,dtype=torch.float32)
    result=torch.empty((rows,blocks.shape[0]),device=x.device,dtype=torch.float32)
    _split[(rows,)](x,hi,lo,factor,width,tr.next_power_of_2(width))
    bm=16 if rows<=16 else min(64,tr.next_power_of_2(rows));wide=rows>=512;bn=128 if wide else 64
    _project[(tr.cdiv(rows,bm),tr.cdiv(blocks.shape[0],bn))](hi,lo,factor,blocks,scales,result,
        rows,blocks.shape[0],width,bm,bn,64,num_warps=8 if wide else 4,num_stages=2 if wide else 3)
    return result
