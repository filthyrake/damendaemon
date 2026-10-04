"""Compensated Tensor Core products for FP32 activations and FP16 weights.

Prepare each input row once, using a power-of-two scale and high/low FP16
components. Both products accumulate in FP32; matrix tiles reuse that prepared
input. This is a numerical approximation, qualified against FP32 references.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _split(X,HI,LO,S,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BLOCK)
    x=tl.load(X+row*K+k,k<K,other=0)
    maximum=tl.max(tl.abs(x),0)
    scale=tl.exp2(tl.maximum(-126.,tl.minimum(120.,tl.floor(tl.log2(maximum))-14.)))
    value=x/scale
    hi=value.to(tl.float16)
    lo=(value-hi.to(tl.float32)).to(tl.float16)
    tl.store(HI+row*K+k,hi,k<K)
    tl.store(LO+row*K+k,lo,k<K)
    tl.store(S+row,scale)


@tr.jit
def _matmul(HI,LO,S,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    scale=tl.load(S+rows,rows<M,other=1.)
    for start in range(tr.cdiv(K,BK)):
        indices=start*BK+k
        hi=tl.load(HI+rows[:,None]*K+indices[None,:],(rows[:,None]<M)&(indices[None,:]<K),other=0)
        lo=tl.load(LO+rows[:,None]*K+indices[None,:],(rows[:,None]<M)&(indices[None,:]<K),other=0)
        w=tl.load(W+cols[None,:]*K+indices[:,None],(cols[None,:]<N)&(indices[:,None]<K),other=0)
        partial=tl.dot(hi,w)+tl.dot(lo,w)
        acc+=partial*scale[:,None]
    result=acc
    tl.store(Y+rows[:,None]*N+cols[None,:],result,(rows[:,None]<M)&(cols[None,:]<N))



def linear(x, weight, *, wide_tiles=True):
    rows, width = x.shape
    hi = torch.empty_like(x, dtype=torch.float16)
    lo = torch.empty_like(hi)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    result = torch.empty((rows, weight.shape[0]), device=x.device, dtype=torch.float32)
    _split[(rows,)](x, hi, lo, scales, width, tr.next_power_of_2(width))
    bm = 16 if rows <= 16 else min(64, tr.next_power_of_2(rows))
    # Wider output tiles amortize loads for long prompts. Keep the original
    # layout for small rows, where fewer blocks can reduce GPU utilization.
    wide = wide_tiles and rows >= 512
    bn = 128 if wide else 64
    _matmul[(tr.cdiv(rows,bm),tr.cdiv(weight.shape[0],bn))](
        hi,lo,scales,weight,result,rows,weight.shape[0],width,bm,bn,64,
        num_warps=8 if wide else 4,num_stages=2 if wide else 3)
    return result
