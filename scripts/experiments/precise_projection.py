"""Experimental projection accumulation policies; never selected by serving."""
import torch
import triton as tr
import triton.language as tl
from damendaemon.precise_matmul import _split


@tr.jit
def _project(X,HI,LO,S,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,MODE:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float64 if MODE=='fp64_tiles' else tl.float32)
    low=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        indices=start*BK+k
        mask=(rows[:,None]<M)&(indices[None,:]<K)
        w=tl.load(W+cols[None,:]*K+indices[:,None],(cols[None,:]<N)&(indices[:,None]<K),other=0)
        if MODE=='tf32x3':
            x=tl.load(X+rows[:,None]*K+indices[None,:],mask,other=0)
            acc=tl.dot(x,w.to(tl.float32),acc,input_precision='tf32x3')
        else:
            hi=tl.load(HI+rows[:,None]*K+indices[None,:],mask,other=0)
            lo=tl.load(LO+rows[:,None]*K+indices[None,:],mask,other=0)
            if MODE=='separate':
                acc=tl.dot(hi,w,acc)
                low=tl.dot(lo,w,low)
            else:
                acc+=tl.dot(hi,w).to(tl.float64)+tl.dot(lo,w).to(tl.float64)
    if MODE!='tf32x3':
        scale=tl.load(S+rows,rows<M,other=1.)
        acc=(acc+low)*scale[:,None]
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def linear(x,weight,mode):
    rows,width=x.shape
    hi=torch.empty_like(x,dtype=torch.float16);lo=torch.empty_like(hi)
    scales=torch.empty(rows,device=x.device,dtype=torch.float32)
    if mode!='tf32x3':_split[(rows,)](x,hi,lo,scales,width,tr.next_power_of_2(width))
    result=torch.empty((rows,weight.shape[0]),device=x.device,dtype=torch.float32)
    bm=16 if rows<=16 else min(64,tr.next_power_of_2(rows));wide=rows>=512
    bn=128 if wide else 64
    _project[(tr.cdiv(rows,bm),tr.cdiv(weight.shape[0],bn))](x,hi,lo,scales,weight,result,
        rows,weight.shape[0],width,bm,bn,64,mode,num_warps=8 if wide else 4,
        num_stages=2 if wide else 3,enable_fp_fusion=False)
    return result
