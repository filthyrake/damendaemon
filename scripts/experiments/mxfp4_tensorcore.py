"""Experiment only: compensated Tensor Core projection of one packed expert route."""
import torch
import triton as tr
import triton.language as tl
from damendaemon.precise_matmul import _split
from damendaemon.mxfp4_kernels import _value_normal as _value


@tr.jit
def _project(HI,LO,SCALE,W,WS,EXPERTS,BIAS,Y,E:tl.constexpr,K:tl.constexpr,N:tl.constexpr,TOP:tl.constexpr,
             SHARED:tl.constexpr,HAS_BIAS:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    route=tl.program_id(1);expert=tl.load(EXPERTS+route)
    valid=(expert>=0)&(expert<E);source=route//TOP if SHARED else route
    rows=tl.arange(0,16);cols=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    scale=tl.load(SCALE+source);acc=tl.full((16,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        indices=start*BK+k
        hi=tl.load(HI+source*K+indices[None,:]+rows[:,None]*0,(rows[:,None]==0)&(indices[None,:]<K),other=0)
        lo=tl.load(LO+source*K+indices[None,:]+rows[:,None]*0,(rows[:,None]==0)&(indices[None,:]<K),other=0)
        mask=valid&(cols[None,:]<N)&(indices[:,None]<K)
        packed=tl.load(W+(expert*N+cols[None,:])*(K//2)+indices[:,None]//2,mask,other=0).to(tl.int32)
        exponent=tl.load(WS+(expert*N+cols[None,:])*(K//32)+indices[:,None]//32,mask,other=127).to(tl.int32)
        code=(packed>>((indices[:,None]&1)*4))&15
        weight=_value(code,exponent).to(tl.float16)
        partial=tl.dot(hi,weight)+tl.dot(lo,weight)
        acc+=partial*scale
    if HAS_BIAS:acc+=tl.load(BIAS+expert*N+cols[None,:],valid&(cols[None,:]<N),other=0)
    acc=tl.where(valid,acc,float('nan'))
    tl.store(Y+route*N+cols[None,:]+rows[:,None]*0,acc,(rows[:,None]==0)&(cols[None,:]<N))


def routed_linear(x,blocks,scales,selected,bias,shared_input,chunked=True,bn=64,bk=64):
    e,n=blocks.shape[:2];k=x.shape[-1]
    hi=torch.empty_like(x,dtype=torch.float16);lo=torch.empty_like(hi)
    scale=torch.empty(x.shape[0],device=x.device,dtype=torch.float32)
    y=torch.empty((selected.numel(),n),device=x.device,dtype=torch.float32)
    _split[(x.shape[0],)](x,hi,lo,scale,k,tr.next_power_of_2(k))
    _project[(tr.cdiv(n,bn),selected.numel())](hi,lo,scale,blocks,scales,selected,
        x if bias is None else bias,y,e,k,n,selected.shape[1],shared_input,bias is not None,bn,bk,
        num_warps=4,num_stages=3)
    return y
