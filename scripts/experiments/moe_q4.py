"""Experimental routed Q4_0 expert projections with FP32 arithmetic."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _routed(X, W, S, EXPERTS, Y, K:tl.constexpr, N:tl.constexpr, TOP:tl.constexpr,
            SHARED:tl.constexpr, BK:tl.constexpr, BN:tl.constexpr):
    route = tl.program_id(1)
    expert = tl.load(EXPERTS+route)
    source = route//TOP if SHARED else route
    outputs = tl.program_id(0)*BN+tl.arange(0,BN)
    columns = tl.arange(0,BK)
    # Each Q4_0 group stores columns 0..15 in low nibbles and 16..31 in high.
    packed = tl.load(W+(expert*N+outputs[:,None])*(K//2)
                     +(columns[None,:]//32)*16+columns[None,:]%16,
                     (outputs[:,None]<N)&(columns[None,:]<K),other=0).to(tl.uint32)
    values = ((packed >> ((columns[None,:]%32//16)*4)) & 15).to(tl.float32)-8
    groups = tl.arange(0,BK//32)
    scales = tl.load(S+(expert*N+outputs[:,None])*(K//32)+groups[None,:],
                     (outputs[:,None]<N)&(groups[None,:]<K//32),other=0).to(tl.float32)
    values = (values.reshape(BN,BK//32,32)*scales[:,:,None]).reshape(BN,BK)
    x = tl.load(X+source*K+columns,columns<K,other=0).to(tl.float32)
    result = tl.sum(values*x[None,:],1)
    tl.store(Y+route*N+outputs,result,outputs<N)


def routed_linear(x, weight, scales, selected, outputs, shared_input):
    # Internal top-k indices must be valid for the supplied expert matrix.
    if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.float32 or x.shape[1]%32
            or selected.ndim != 2 or selected.dtype != torch.int64 or selected.shape[1] < 1):
        raise ValueError('CUDA FP32 inputs and nonempty int64 routes required')
    if outputs < 1 or weight.ndim != 2 or weight.shape[0]%outputs or weight.shape[1]*2 != x.shape[1]:
        raise ValueError('invalid Q4 expert matrix geometry')
    rows = selected.shape[0] if shared_input else selected.numel()
    if x.shape[0] != rows or not selected.numel():
        raise ValueError('expert input rows disagree with routes')
    if (weight.dtype != torch.uint8 or scales.dtype != torch.float16
            or scales.shape != (weight.shape[0],x.shape[1]//32)
            or any(value.device != x.device for value in (weight,scales,selected))
            or not weight.is_contiguous() or not scales.is_contiguous()):
        raise ValueError('contiguous same-device Q4 storage and FP16 scales required')
    x, selected = x.contiguous(), selected.contiguous()
    result = torch.empty((selected.numel(),outputs),device=x.device,dtype=x.dtype)
    block = tr.next_power_of_2(x.shape[-1])
    tile = 4 if block>=2048 else 8
    _routed[(tr.cdiv(outputs,tile),selected.numel())](x,weight,scales,selected,result,
        x.shape[-1],outputs,selected.shape[-1],shared_input,block,tile,
        num_warps=4,enable_fp_fusion=False)
    return result
