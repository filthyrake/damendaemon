"""Native MXFP4 expert reads and reductions, independent of reference runtimes."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _value(code,scale):
    unsigned=code & 7
    magnitude=tl.where(unsigned<4,unsigned.to(tl.float32)*.5,
        tl.where(unsigned<6,unsigned.to(tl.float32)-2.,2.*unsigned.to(tl.float32)-8.))
    # E8M0 is the FP32 exponent field, except exponent zero represents
    # 2**-127 rather than zero. The multiplication is exact until overflow:
    # E2M1 has only two significant bits, including at the subnormal boundary.
    scale_bits=tl.where(scale==0,0x00400000,scale.to(tl.int32)<<23)
    scaled=magnitude*scale_bits.to(tl.float32,bitcast=True)
    bits=scaled.to(tl.int32,bitcast=True) | ((code & 8)<<28)
    bits=tl.where(scale==255,0x7fc00000,bits)
    return bits.to(tl.float32,bitcast=True)



@tr.jit
def _unpack(W,S,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    packed=tl.load(W+index//2,index<N,other=0).to(tl.int32)
    code=(packed >> ((index & 1)*4)) & 15
    scale=tl.load(S+index//32,index<N,other=127).to(tl.int32)
    tl.store(Y+index,_value(code,scale),index<N)


def unpack(blocks,scales,shape,dtype):
    result=torch.empty(shape,device=blocks.device,dtype=dtype)
    _unpack[(tr.cdiv(result.numel(),256),)](blocks,scales,result,result.numel(),256,
        enable_fp_fusion=False)
    return result


@tr.jit
def _routed(X,W,S,EXPERTS,BIAS,Y,E:tl.constexpr,K:tl.constexpr,N:tl.constexpr,TOP:tl.constexpr,
            SHARED:tl.constexpr,HAS_BIAS:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
    route=tl.program_id(1)
    expert=tl.load(EXPERTS+route)
    valid_expert=(expert>=0)&(expert<E)
    source=route//TOP if SHARED else route
    outputs=tl.program_id(0)*BN+tl.arange(0,BN)
    local=tl.arange(0,BK//2)
    row=expert*N+outputs[:,None]
    accumulator=tl.full((BN,BK//2),0,tl.float32)
    for offset in range(tl.cdiv(K,BK)):
        pairs=offset*(BK//2)+local
        valid=valid_expert & (outputs[:,None]<N) & (pairs[None,:]<K//2)
        packed=tl.load(W+row*(K//2)+pairs[None,:],valid,other=0).to(tl.int32)
        scale=tl.load(S+row*(K//32)+pairs[None,:]//16,valid,other=127).to(tl.int32)
        low=_value(packed & 15,scale)
        high=_value(packed >> 4,scale)
        even=tl.load(X+source*K+2*pairs,pairs<K//2,other=0).to(tl.float32)
        odd=tl.load(X+source*K+2*pairs+1,pairs<K//2,other=0).to(tl.float32)
        accumulator+=low*even[None,:]+high*odd[None,:]
    result=tl.sum(accumulator,1)
    if HAS_BIAS:
        result+=tl.load(BIAS+expert*N+outputs,valid_expert & (outputs<N),other=0)
    # Internal router indices are bounded by topk; retain memory safety if a
    # faulty caller supplies an invalid index, and make failure conspicuous.
    result=tl.where(valid_expert,result,float('nan'))
    tl.store(Y+route*N+outputs,result,outputs<N)


def routed_linear(x,blocks,scales,selected,bias,shared_input,chunked=True):
    experts,outputs=blocks.shape[:2]
    result=torch.empty((selected.numel(),outputs),device=x.device,dtype=torch.float32)
    block=tr.next_power_of_2(x.shape[-1])
    tile=4
    if chunked and x.shape[-1]>=2048:
        block=512
        tile=4 if shared_input else 8
    _routed[(tr.cdiv(outputs,tile),selected.numel())](x,blocks,scales,selected,
        x if bias is None else bias,result,experts,x.shape[-1],outputs,selected.shape[1],
        shared_input,bias is not None,block,tile,num_warps=4,enable_fp_fusion=False)
    return result


@tr.jit
def _swiglu(X,Y,N:tl.constexpr,LIMIT:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    gate=tl.load(X+2*index,index<N,other=0)
    up=tl.load(X+2*index+1,index<N,other=0)
    gate=tl.minimum(gate,LIMIT)
    up=tl.minimum(tl.maximum(up,-LIMIT),LIMIT)
    value=gate*tl.sigmoid(1.702*gate)*(up+1.)
    tl.store(Y+index,value,index<N)


def swiglu(x,limit=7.):
    if (x.device.type!='cuda' or x.dtype!=torch.float32 or x.ndim!=2
            or x.shape[1]%2 or not x.is_contiguous()):
        raise ValueError('GPT-OSS activation requires contiguous CUDA FP32 interleaved gate/up rows')
    result=torch.empty((x.shape[0],x.shape[1]//2),device=x.device,dtype=x.dtype)
    _swiglu[(tr.cdiv(result.numel(),256),)](x,result,result.numel(),limit,256,enable_fp_fusion=False)
    return result
