"""Bounded compensated FP32 GQA prefill; experimental until full-model gates.

Each program keeps one query tile and streams KV tiles. No expanded KV cache
or context-sized score matrix. Three BF16 products approximate FP32 products,
matching the existing projection precision strategy, not exact IEEE arithmetic.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _dot(a, b):
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    al = (a-ah.to(tl.float32)).to(tl.bfloat16)
    bl = (b-bh.to(tl.float32)).to(tl.bfloat16)
    low = tl.dot(ah, bl)
    low = tl.dot(al, bh, low)
    return tl.dot(ah, bh) + low


@tr.jit
def _attention(Q,K,V,O, QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
               KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
               VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,
               H:tl.constexpr,KH:tl.constexpr,QCOUNT:tl.constexpr,KCOUNT:tl.constexpr,
               D:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    block,head,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    rows=block*BM+tl.arange(0,BM)
    dims=tl.arange(0,BD)
    q=tl.load(Q+batch*QS0+head*QS1+rows[:,None]*QS2+dims[None,:],
              (rows[:,None]<QCOUNT)&(dims[None,:]<D),other=0).to(tl.float32)
    maximum=tl.full((BM,),-float('inf'),tl.float32)
    denominator=tl.full((BM,),0.,tl.float32)
    accumulator=tl.full((BM,BD),0.,tl.float32)
    prefix=KCOUNT-QCOUNT
    end=tl.minimum(KCOUNT,prefix+(block+1)*BM)
    kvhead=head//(H//KH)
    for start in range(tl.cdiv(end,BN)):
        cols=start*BN+tl.arange(0,BN)
        k=tl.load(K+batch*KS0+kvhead*KS1+cols[None,:]*KS2+dims[:,None],
                  (cols[None,:]<KCOUNT)&(dims[:,None]<D),other=0).to(tl.float32)
        scores=_dot(q,k)*(D**-.5)
        scores=tl.where((cols[None,:]<=prefix+rows[:,None])&(cols[None,:]<KCOUNT),scores,-float('inf'))
        next_max=tl.maximum(maximum,tl.max(scores,1))
        correction=tl.exp(maximum-next_max)
        probabilities=tl.exp(scores-next_max[:,None])
        denominator=denominator*correction+tl.sum(probabilities,1)
        v=tl.load(V+batch*VS0+kvhead*VS1+cols[:,None]*VS2+dims[None,:],
                  (cols[:,None]<KCOUNT)&(dims[None,:]<D),other=0).to(tl.float32)
        accumulator=accumulator*correction[:,None]+_dot(probabilities,v)
        maximum=next_max
    output=accumulator/denominator[:,None]
    tl.store(O+((batch*H+head)*QCOUNT+rows[:,None])*D+dims[None,:],output,
             (rows[:,None]<QCOUNT)&(dims[None,:]<D))


def attention(q,k,v,**kwargs):
    if (q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0]
            or q.shape[-1]!=k.shape[-1] or q.shape[1]%k.shape[1]
            or not 0<q.shape[2]<=k.shape[2] or q.shape[-1] not in (128,256)
            or any(x.dtype!=torch.float32 or x.device!=q.device or x.stride(-1)!=1 for x in (q,k,v))
            or q.device.type!='cuda'):
        raise ValueError('requires FP32 CUDA/ROCm GQA causal tensors with contiguous head dimensions')
    output=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    _attention[(tr.cdiv(q.shape[2],16),q.shape[1],q.shape[0])](q,k,v,output,
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],q.shape[1],k.shape[1],q.shape[2],k.shape[2],
        q.shape[-1],tr.next_power_of_2(q.shape[-1]),16,32,num_warps=4,num_stages=1)
    return output
