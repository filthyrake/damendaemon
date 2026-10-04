"""Independent FP32 Mamba convolution and selective recurrence kernels."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _conv(X,W,BIAS,HISTORY,SLOTS,Y,I:tl.constexpr,T:tl.constexpr,C:tl.constexpr,STRIDE:tl.constexpr,RESET:tl.constexpr,BLOCK:tl.constexpr):
    row,block,t=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    channels=block*BLOCK+tl.arange(0,BLOCK);valid=channels<I
    slot=tl.load(SLOTS+row);acc=tl.load(BIAS+channels,valid,other=0)
    for j in tl.static_range(C):
        if RESET:
            source=t+j-C+1
            value=tl.load(X+(row*T+source)*STRIDE+channels,valid&(source>=0),other=0)
        else:
            if j<C-1:
                value=tl.load(HISTORY+(slot*I+channels)*C+j+1,valid,other=0)
                tl.store(HISTORY+(slot*I+channels)*C+j,value,valid)
            else:
                value=tl.load(X+row*STRIDE+channels,valid,other=0)
                tl.store(HISTORY+(slot*I+channels)*C+j,value,valid)
        acc+=value*tl.load(W+channels*C+j,valid,other=0)
    tl.store(Y+(row*T+t)*I+channels,acc/(1+tl.exp(-acc)),valid)
    if RESET:
        if t==T-1:
            for j in tl.static_range(C):
                source=T-C+j
                value=tl.load(X+(row*T+source)*STRIDE+channels,valid&(source>=0),other=0)
                tl.store(HISTORY+(slot*I+channels)*C+j,value,valid)


@tr.jit
def _conv_continue(X,W,BIAS,HISTORY,SLOTS,Y,I:tl.constexpr,T:tl.constexpr,C:tl.constexpr,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row,block=tl.program_id(0),tl.program_id(1)
    channels=block*BLOCK+tl.arange(0,BLOCK);valid=channels<I
    slot=tl.load(SLOTS+row)
    # One program owns each channel's history. Keep the old history intact
    # until every output that can reference it has been computed.
    for t in range(T):
        acc=tl.load(BIAS+channels,valid,other=0)
        for j in tl.static_range(C):
            source=t+j-C+1
            if source<0:
                value=tl.load(HISTORY+(slot*I+channels)*C+source+C,valid,other=0)
            else:
                value=tl.load(X+(row*T+source)*STRIDE+channels,valid,other=0)
            acc+=value*tl.load(W+channels*C+j,valid,other=0)
        tl.store(Y+(row*T+t)*I+channels,acc/(1+tl.exp(-acc)),valid)
    for j in tl.static_range(C):
        source=T-C+j
        if source<0:
            value=tl.load(HISTORY+(slot*I+channels)*C+source+C,valid,other=0)
        else:
            value=tl.load(X+(row*T+source)*STRIDE+channels,valid,other=0)
        tl.store(HISTORY+(slot*I+channels)*C+j,value,valid)


def conv(x,weight,bias,history,slots,reset):
    batch,length,stride=x.shape;inner=history.shape[1]
    out=torch.empty((batch,length,inner),device=x.device,dtype=x.dtype)
    if not reset and length>1:
        _conv_continue[(batch,tr.cdiv(inner,128))](x,weight,bias,history,slots,out,inner,length,weight.shape[-1],stride,128,enable_fp_fusion=False)
    else:
        _conv[(batch,tr.cdiv(inner,128),length)](x,weight,bias,history,slots,out,inner,length,weight.shape[-1],stride,reset,128,enable_fp_fusion=False)
    return out


@tr.jit
def _scan(X,G,P,DT,BIAS,A,D,STATE,SLOTS,Y,I:tl.constexpr,T:tl.constexpr,S:tl.constexpr,R:tl.constexpr,
          GSTRIDE:tl.constexpr,RESET:tl.constexpr,BC:tl.constexpr,BS:tl.constexpr):
    row,block=tl.program_id(0),tl.program_id(1)
    channels=block*BC+tl.arange(0,BC);states=tl.arange(0,BS)
    slot=tl.load(SLOTS+row);mask=(channels[:,None]<I)&(states[None,:]<S)
    ptr=STATE+(slot*I+channels[:,None])*S+states[None,:]
    value=tl.full((BC,BS),0,tl.float32) if RESET else tl.load(ptr,mask,other=0)
    decay=tl.load(A+channels[:,None]*S+states[None,:],mask,other=0)
    bias=tl.load(BIAS+channels,channels<I,other=0);skip=tl.load(D+channels,channels<I,other=0)
    for t in range(T):
        source=(row*T+t)*I+channels
        dt=tl.load(DT+source,channels<I,other=0)+bias
        dt=tl.maximum(dt,0)+libdevice.log1p(tl.exp(-tl.abs(dt)))
        signal=tl.load(X+source,channels<I,other=0)
        b=tl.load(P+(row*T+t)*(R+2*S)+R+states,states<S,other=0)
        c=tl.load(P+(row*T+t)*(R+2*S)+R+S+states,states<S,other=0)
        value=tl.exp(dt[:,None]*decay)*value+(dt[:,None]*b[None,:])*signal[:,None]
        result=tl.sum(value*c[None,:],1)+skip*signal
        gate=tl.load(G+(row*T+t)*GSTRIDE+GSTRIDE-I+channels,channels<I,other=0)
        tl.store(Y+source,result*(gate/(1+tl.exp(-gate))),channels<I)
    tl.store(ptr,value,mask)


def scan(x,gate,params,dt,bias,a,d,state,slots,reset):
    batch,length,inner=x.shape;s=state.shape[-1];rank=params.shape[-1]-2*s
    out=torch.empty_like(x)
    _scan[(batch,tr.cdiv(inner,16))](x,gate,params,dt,bias,a,d,state,slots,out,inner,length,s,rank,gate.shape[-1],reset,16,tr.next_power_of_2(s),num_warps=4,enable_fp_fusion=False)
    return out
