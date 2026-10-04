"""Fused CUDA log-normalization and bounded top-k, with one result readback."""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _parts(X,STATS,VALUES,IDS,N:tl.constexpr,PARTS:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    row,part=tl.program_id(0),tl.program_id(1)
    columns=part*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(X+row*N+columns,columns<N,other=-float('inf')).to(tl.float32)
    maximum=tl.max(values,0)
    shifted=tl.where(maximum==-float('inf'),-float('inf'),values-maximum)
    total=tl.sum(tl.exp(shifted),0)
    tl.store(STATS+2*(row*PARTS+part),maximum)
    tl.store(STATS+2*(row*PARTS+part)+1,total)
    for rank in range(K):
        best=tl.max(values,0)
        token=tl.min(tl.where((values==best)&(columns<N),columns,2147483647),0)
        offset=(row*PARTS+part)*K+rank
        tl.store(VALUES+offset,best)
        tl.store(IDS+offset,token)
        values=tl.where(columns==token,-float('inf'),values)


@tr.jit
def _finish(X,SELECTED,STATS,VALUES,IDS,OUT,N:tl.constexpr,PARTS:tl.constexpr,K:tl.constexpr,
            BP:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0)
    parts=tl.arange(0,BP)
    maxima=tl.load(STATS+2*(row*PARTS+parts),parts<PARTS,other=-float('inf'))
    totals=tl.load(STATS+2*(row*PARTS+parts)+1,parts<PARTS,other=0.)
    maximum=tl.max(maxima,0)
    normalizer=tl.log(tl.sum(totals*tl.exp(maxima-maximum),0))
    selected=tl.load(SELECTED+row)
    value=tl.load(X+row*N+selected).to(tl.float32)
    base=row*(K+1)*2
    tl.store(OUT+base,selected.to(tl.float32))
    tl.store(OUT+base+K+1,(value-maximum)-normalizer)
    if K:
        offsets=tl.arange(0,BC)
        values=tl.load(VALUES+row*PARTS*K+offsets,offsets<PARTS*K,other=-float('inf'))
        ids=tl.load(IDS+row*PARTS*K+offsets,offsets<PARTS*K,other=2147483647)
        for rank in range(K):
            best=tl.max(values,0)
            token=tl.min(tl.where(values==best,ids,2147483647),0)
            tl.store(OUT+base+rank+1,token.to(tl.float32))
            tl.store(OUT+base+K+rank+2,(best-maximum)-normalizer)
            values=tl.where(ids==token,-float('inf'),values)


def score_rows(logits,selected,tops):
    rows,vocab=logits.shape
    if vocab>262144:
        raise ValueError('native scoring supports vocabularies up to 262144 tokens')
    k=min(max(tops),vocab)
    parts=tr.cdiv(vocab,1024)
    logits=logits.contiguous();selected=selected.contiguous()
    stats=torch.empty((rows,parts,2),device=logits.device,dtype=torch.float32)
    values=torch.empty((rows,parts,k),device=logits.device,dtype=torch.float32)
    ids=torch.empty((rows,parts,k),device=logits.device,dtype=torch.int32)
    result=torch.empty((rows,2*(k+1)),device=logits.device,dtype=torch.float32)
    _parts[(rows,parts)](logits,stats,values,ids,vocab,parts,k,1024,enable_fp_fusion=False)
    _finish[(rows,)](logits,selected,stats,values,ids,result,vocab,parts,k,
        tr.next_power_of_2(parts),tr.next_power_of_2(max(1,parts*k)),enable_fp_fusion=False)
    packets=[]
    for row,top in zip(result.cpu().tolist(),tops):
        probabilities=row[k+1:]
        if any(not math.isfinite(value) for value in probabilities):
            raise ValueError('cannot report non-finite model log probabilities')
        packets.append({'token_id':int(row[0]),'logprob':probabilities[0],
            'top':[(int(row[i+1]),probabilities[i+1]) for i in range(min(top,k))]})
    return packets
