"""CUDA Q4 body products over adjacent-nibble stage storage."""
import triton as tr
import triton.language as tl


@tr.jit
def single_dot_adjacent(X,W,S,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
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
        w=tl.join(low,high).reshape(BN,BK)
        accum+=tl.dot(x,tl.trans(w))
    if BIASED:accum+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],accum,(rows[:,None]<M)&(outputs[None,:]<N))


@tr.jit
def _expand_adjacent(W,S,Y,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,col=offsets//K,offsets%K
    packed=tl.load(W+row*(K//2)+col//2,offsets<N*K,other=0).to(tl.int32)
    value=((packed>>((col%2)*4))&15)-8
    scale=tl.load(S+row*(K//32)+col//32,offsets<N*K,other=0).to(tl.float32)
    tl.store(Y+offsets,value.to(tl.float32)*scale,offsets<N*K)


def expand_adjacent(weight,scales,dtype):
    import torch
    n,k=weight.shape[0],weight.shape[1]*2
    output=torch.empty((n,k),device=weight.device,dtype=dtype)
    _expand_adjacent[(tr.cdiv(n*k,1024),)](weight,scales,output,n,k,1024)
    return output
