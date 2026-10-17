"""GPU-selected expert projections; routing decisions never leave the device."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _routed(X, W, S, EXPERTS, Y, K:tl.constexpr, N:tl.constexpr, TOP:tl.constexpr,
            SHARED:tl.constexpr, QUANTIZED:tl.constexpr, BK:tl.constexpr, BN:tl.constexpr):
    route = tl.program_id(1)
    expert = tl.load(EXPERTS+route)
    source = route//TOP if SHARED else route
    outputs = tl.program_id(0)*BN+tl.arange(0,BN)
    columns = tl.arange(0,BK)
    values = tl.load(W+(expert*N+outputs[:,None])*K+columns[None,:],
        (outputs[:,None]<N)&(columns[None,:]<K),other=0).to(tl.float32)
    if QUANTIZED:
        groups = tl.arange(0,BK//32)
        scales = tl.load(S+(expert*N+outputs[:,None])*(K//32)+groups[None,:],
            (outputs[:,None]<N)&(groups[None,:]<K//32),other=0).to(tl.float32)
        values = (values.reshape(BN,BK//32,32)*scales[:,:,None]).reshape(BN,BK)
    values = values.to(X.dtype.element_ty).to(tl.float32)
    x = tl.load(X+source*K+columns,columns<K,other=0).to(tl.float32)
    result = tl.sum(values*x[None,:],1)
    tl.store(Y+route*N+outputs,result,outputs<N)


def routed_linear(x, weight, scales, selected, outputs, shared_input):
    x = x.contiguous()
    selected = selected.contiguous()
    result = torch.empty((selected.numel(),outputs),device=x.device,dtype=x.dtype)
    block = tr.next_power_of_2(x.shape[-1])
    tile = 4 if block>=2048 else 8
    _routed[(tr.cdiv(outputs,tile),selected.numel())](x,weight,weight if scales is None else scales,
        selected,result,x.shape[-1],outputs,selected.shape[-1],shared_input,scales is not None,
        block,tile,num_warps=4,enable_fp_fusion=False)
    return result


@tr.jit
def _combine(X, P, Y, H:tl.constexpr, TOP:tl.constexpr, BT:tl.constexpr, BLOCK:tl.constexpr):
    token = tl.program_id(0)
    columns = tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    routes = tl.arange(0,BT)
    values = tl.load(X+(token*TOP+routes[:,None])*H+columns[None,:],
        (routes[:,None]<TOP)&(columns[None,:]<H),other=0).to(tl.float32)
    probabilities = tl.load(P+token*TOP+routes,routes<TOP,other=0).to(tl.float32)
    result = tl.sum(values*probabilities[:,None],0)
    tl.store(Y+token*H+columns,result,columns<H)


def combine(projected, probabilities):
    hidden = projected.shape[-1]
    result = torch.empty((probabilities.shape[0],hidden),device=projected.device,dtype=projected.dtype)
    _combine[(result.shape[0],tr.cdiv(hidden,128))](projected,probabilities,result,
        hidden,probabilities.shape[-1],tr.next_power_of_2(probabilities.shape[-1]),128)
    return result
