"""FP32 input/output projections using TF32x3 Tensor Core decomposition."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _matmul(X,W,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    accumulator=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+columns
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        weight=tl.load(W+outputs[None,:]*K+k[:,None],(outputs[None,:]<N)&(k[:,None]<K),other=0)
        accumulator+=tl.dot(x,weight,input_precision='tf32x3')
    if BIASED:accumulator+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:]
    tl.store(Y+rows[:,None]*N+outputs[None,:],accumulator,(rows[:,None]<M)&(outputs[None,:]<N))


def linear(x,weight,bias=None):
    if x.dtype!=torch.float32 or weight.dtype!=torch.float32:
        raise ValueError('TF32x3 projection requires FP32 inputs and weights')
    shape=x.shape[:-1];x=x.reshape(-1,x.shape[-1]).contiguous()
    result=torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=torch.float32)
    _matmul[(tr.cdiv(x.shape[0],32),tr.cdiv(weight.shape[0],64))](x,weight,
        x if bias is None else bias,result,x.shape[0],weight.shape[0],weight.shape[1],32,64,32,bias is not None,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return result.reshape(*shape,weight.shape[0])
