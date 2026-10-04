"""Native CUDA matrix products over Q4_0 nibbles and FP16 block scales."""
import torch
import triton as tr
import triton.language as tl


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
          BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    outputs=tl.program_id(0)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    weight=_values(W,S,outputs[:,None],columns[None,:],N,K).to(X.dtype.element_ty).to(tl.float32)
    for row in range(ROWS):
        x=tl.load(X+row*K+columns,columns<K,other=0).to(tl.float32)
        value=tl.sum(weight*x[None,:],1)
        if BIASED:value+=tl.load(BIAS+outputs,outputs<N,other=0).to(tl.float32)
        tl.store(Y+row*N+outputs,value,outputs<N)


@tr.jit
def _gemm(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    result=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+columns
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        weight=_values(W,S,outputs[None,:],k[:,None],N,K).to(tl.float16)
        result+=tl.dot(x,weight)
    if BIASED:result+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],result,(rows[:,None]<M)&(outputs[None,:]<N))


def linear(x,weight,scales,bias=None):
    shape=x.shape[:-1]
    x=x.reshape(-1,x.shape[-1]).contiguous()
    rows,k=x.shape;n=weight.shape[0]
    if weight.shape[1]*2!=k:raise ValueError('Q4 matrix width mismatch')
    if x.dtype==torch.float16 and 2<=rows<=128:
        result=torch.empty((rows,n),device=x.device,dtype=x.dtype)
        bm=16 if rows<=16 else tr.next_power_of_2(rows)
        bk=128 if rows<=16 else 64
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
