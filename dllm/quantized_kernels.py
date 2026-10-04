"""CUDA Q8_0 matrix-vector arithmetic; no resident dequantized weight copy."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _expand(W, S, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    groups = tl.program_id(0) * (BLOCK//32) + tl.arange(0, BLOCK//32)
    offsets = groups[:,None] * 32 + tl.arange(0,32)[None,:]
    values = tl.load(W + offsets, offsets < TOTAL, other=0).to(tl.float32)
    scales = tl.load(S + groups, groups < TOTAL//32, other=0).to(tl.float32)
    tl.store(Y + offsets, values * scales[:,None], offsets < TOTAL)


def expand(weight, scales, dtype):
    result = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    _expand[(tr.cdiv(weight.numel(),1024),)](weight, scales, result, weight.numel(), 1024)
    return result


@tr.jit
def _matmul(X, W, S, BIAS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BIASED: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0,BM)
    outputs = tl.program_id(1)*BN + tl.arange(0,BN)
    columns = tl.arange(0,BK)
    groups = tl.arange(0,BK//32)
    accumulator = tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k = start*BK+columns
        x = tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        values = tl.load(W+outputs[None,:]*K+k[:,None],(outputs[None,:]<N)&(k[:,None]<K),other=0).to(tl.float32)
        g = start*(BK//32)+groups
        scales = tl.load(S+outputs[None,:]*(K//32)+g[:,None],(outputs[None,:]<N)&(g[:,None]<K//32),other=0).to(tl.float32)
        weight = (values.reshape(BK//32,32,BN)*scales[:,None,:]).reshape(BK,BN).to(tl.float16)
        accumulator += tl.dot(x,weight)
    if BIASED:
        accumulator += tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],accumulator,(rows[:,None]<M)&(outputs[None,:]<N))


def matmul(x, weight, scales, bias=None):
    result = torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
    _matmul[(tr.cdiv(x.shape[0],16),tr.cdiv(weight.shape[0],64))](x,weight,scales,
        weight if bias is None else bias,result,x.shape[0],weight.shape[0],weight.shape[1],
        16,64,128,bias is not None,num_warps=4,num_stages=3)
    return result


@tr.jit
def _linear(X, W, S, BIAS, Y, K: tl.constexpr, N: tl.constexpr,
            BK: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, BIASED: tl.constexpr):
    outputs = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    quantized = tl.load(W + outputs[:,None]*K + columns[None,:],
        (outputs[:,None] < N) & (columns[None,:] < K), other=0).to(tl.float32)
    groups = tl.arange(0, BK//32)
    scale = tl.load(S + outputs[:,None]*(K//32) + groups[None,:],
        (outputs[:,None] < N) & (groups[None,:] < K//32), other=0).to(tl.float32)
    # Match the executor's matrix precision after dequantization.
    weight = (quantized.reshape(BN, BK//32, 32) * scale[:,:,None]).reshape(BN, BK).to(X.dtype.element_ty).to(tl.float32)
    for row in range(ROWS):
        x = tl.load(X + row*K + columns, columns < K, other=0).to(tl.float32)
        result = tl.sum(weight * x[None,:], 1)
        if BIASED:
            result += tl.load(BIAS + outputs, outputs < N, other=0).to(tl.float32)
        tl.store(Y + row*N + outputs, result, outputs < N)


def linear(x, weight, scales, bias=None):
    shape = x.shape[:-1]
    x = x.reshape(-1,x.shape[-1]).contiguous()
    if x.dtype == torch.float16 and 2 <= x.shape[0] <= 16:
        result = matmul(x,weight,scales,bias)
    elif x.shape[0] > 4:
        result = torch.nn.functional.linear(x, expand(weight, scales, x.dtype), bias)
    else:
        result = torch.empty((x.shape[0],weight.shape[0]), device=x.device, dtype=x.dtype)
        block = tr.next_power_of_2(x.shape[1])
        outputs = 2 if block >= 4096 else 8
        _linear[(tr.cdiv(weight.shape[0],outputs),)](x,weight,scales,
            weight if bias is None else bias,result,x.shape[1],weight.shape[0],block,outputs,x.shape[0],bias is not None,
            num_warps=4, enable_fp_fusion=False)
    return result.reshape(*shape,weight.shape[0])
