"""Native CUDA matrix products over Q4_0 nibbles and FP16 block scales."""
import torch
import triton as tr
import triton.language as tl
from .q4_prefill_policy import use_small_tile


@tr.jit
def _values(W,S,outputs,columns,N:tl.constexpr,K:tl.constexpr):
    packed=tl.load(W+outputs*(K//2)+(columns//32)*16+columns%16,
                   (outputs<N)&(columns<K),other=0).to(tl.int32)
    shift=tl.where(columns%32<16,0,4)
    values=((packed>>shift)&15)-8
    scales=tl.load(S+outputs*(K//32)+columns//32,(outputs<N)&(columns<K),other=0).to(tl.float32)
    return values.to(tl.float32)*scales


@tr.jit
def _expand(W,S,Y,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    values=_values(W,S,offsets//K,offsets%K,N,K)
    tl.store(Y+offsets,values,offsets<N*K)


def expand(weight,scales,dtype):
    n,k=weight.shape[0],weight.shape[1]*2
    result=torch.empty((n,k),device=weight.device,dtype=dtype)
    _expand[(tr.cdiv(n*k,1024),)](weight,scales,result,n,k,1024)
    return result


@tr.jit
def _gemv(X,W,S,BIAS,Y,N:tl.constexpr,K:tl.constexpr,ROWS:tl.constexpr,
          BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr,PARTS:tl.constexpr=1):
    outputs=tl.program_id(0)*BN+tl.arange(0,BN)
    part=tl.program_id(1)
    groups=part*(BK//32)+tl.arange(0,BK//32)
    half=tl.arange(0,16)
    packed=tl.load(W+outputs[:,None,None]*(K//2)+groups[None,:,None]*16+half[None,None,:],
        (outputs[:,None,None]<N)&(groups[None,:,None]<K//32),other=0).to(tl.int32)
    scale=tl.load(S+outputs[:,None]*(K//32)+groups[None,:],
        (outputs[:,None]<N)&(groups[None,:]<K//32),other=0).to(tl.float32)
    low=(((packed&15)-8).to(tl.float32)*scale[:,:,None]).to(X.dtype.element_ty).to(tl.float32)
    high=(((packed>>4)-8).to(tl.float32)*scale[:,:,None]).to(X.dtype.element_ty).to(tl.float32)
    columns=groups[:,None]*32+half[None,:]
    for row in range(ROWS):
        xlow=tl.load(X+row*K+columns,columns<K,other=0).to(tl.float32)
        xhigh=tl.load(X+row*K+columns+16,columns+16<K,other=0).to(tl.float32)
        products=(low*xlow[None,:,:]+high*xhigh[None,:,:]).reshape(BN,BK//2)
        value=tl.sum(products,1)
        if BIASED:value+=tl.load(BIAS+outputs,outputs<N,other=0).to(tl.float32)
        tl.store(Y+(row*N+outputs)*PARTS+part,value,outputs<N)


@tr.jit
def _gemm(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    packed_columns=tl.arange(0,BK//2)
    groups=tl.arange(0,BK//32)
    result=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+(packed_columns//16)*32+packed_columns%16
        xlow=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        xhigh=tl.load(X+rows[:,None]*K+k[None,:]+16,(rows[:,None]<M)&(k[None,:]+16<K),other=0)
        p=start*(BK//2)+packed_columns
        packed=tl.load(W+outputs[:,None]*(K//2)+p[None,:],(outputs[:,None]<N)&(p[None,:]<K//2),other=0).to(tl.int32)
        g=start*(BK//32)+groups
        scales=tl.load(S+outputs[:,None]*(K//32)+g[None,:],(outputs[:,None]<N)&(g[None,:]<K//32),other=0).to(tl.float32)
        low=(((packed&15)-8).to(tl.float32).reshape(BN,BK//32,16)*scales[:,:,None]).reshape(BN,BK//2).to(tl.float16)
        high=(((packed>>4)-8).to(tl.float32).reshape(BN,BK//32,16)*scales[:,:,None]).reshape(BN,BK//2).to(tl.float16)
        result+=tl.dot(xlow,tl.trans(low))
        result+=tl.dot(xhigh,tl.trans(high))
    if BIASED:result+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],result,(rows[:,None]<M)&(outputs[None,:]<N))


def linear(x,weight,scales,bias=None):
    shape=x.shape[:-1]
    x=x.reshape(-1,x.shape[-1]).contiguous()
    rows,k=x.shape;n=weight.shape[0]
    if weight.shape[1]*2!=k:raise ValueError('Q4 matrix width mismatch')
    if x.dtype==torch.float16 and rows==1 and k>=4096 and n>=1024:
        from .native_q4 import linear_if_enabled
        native=linear_if_enabled(x,weight,scales,bias)
        if native is not None:return native.reshape(*shape,n)
    if x.dtype==torch.float16 and 1<=rows<=128:
        result=torch.empty((rows,n),device=x.device,dtype=x.dtype)
        bm=16 if rows<=16 else tr.next_power_of_2(rows)
        if 32<rows<=64 and use_small_tile(rows,n,k,x.device):bm=32
        bk=128
        _gemm[(tr.cdiv(rows,bm),tr.cdiv(n,64))](x,weight,scales,weight if bias is None else bias,
            result,rows,n,k,bm,64,bk,bias is not None,num_warps=8 if bm==128 else 4,num_stages=3)
    elif rows>4:
        result=torch.nn.functional.linear(x,expand(weight,scales,x.dtype),bias)
    else:
        result=torch.empty((rows,n),device=x.device,dtype=x.dtype)
        bk=tr.next_power_of_2(k);bn=2 if bk>=4096 else 8
        _gemv[(tr.cdiv(n,bn),)](x,weight,scales,weight if bias is None else bias,result,n,k,rows,bn,bk,
                              bias is not None,num_warps=4,enable_fp_fusion=False)
    return result.reshape(*shape,n)
