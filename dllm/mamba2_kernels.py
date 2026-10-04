"""Independent FP32 head-wise Mamba-2 selective recurrence."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _scan(X,G,DT,BIAS,A,D,STATE,SLOTS,Y,I:tl.constexpr,T:tl.constexpr,S:tl.constexpr,H:tl.constexpr,
          RESET:tl.constexpr,BC:tl.constexpr,BS:tl.constexpr):
    row,block=tl.program_id(0),tl.program_id(1)
    channels=block*BC+tl.arange(0,BC);states=tl.arange(0,BS);heads=channels//H
    slot=tl.load(SLOTS+row);mask=(channels[:,None]<I)&(states[None,:]<S)
    address=STATE+(slot*I+channels[:,None])*S+states[None,:]
    value=tl.full((BC,BS),0,tl.float32) if RESET else tl.load(address,mask,other=0)
    decay=tl.load(A+heads,channels<I,other=0)
    bias=tl.load(BIAS+heads,channels<I,other=0);skip=tl.load(D+heads,channels<I,other=0)
    for t in range(T):
        source=(row*T+t)*(I+2*S)
        dt=tl.load(DT+(row*T+t)*(I//H)+heads,channels<I,other=0)+bias
        dt=tl.maximum(dt,0)+libdevice.log1p(tl.exp(-tl.abs(dt)))
        x=tl.load(X+source+channels,channels<I,other=0)
        b=tl.load(X+source+I+states,states<S,other=0)
        c=tl.load(X+source+I+S+states,states<S,other=0)
        value=tl.exp(dt[:,None]*decay[:,None])*value+(dt[:,None]*b[None,:])*x[:,None]
        result=tl.sum(value*c[None,:],1)+skip*x
        gate=tl.load(G+(row*T+t)*I+channels,channels<I,other=0)
        tl.store(Y+(row*T+t)*I+channels,result*(gate/(1+tl.exp(-gate))),channels<I)
    tl.store(address,value,mask)


def scan(x,gate,dt,bias,a,d,state,slots,head_dim,reset):
    batch,length,inner=gate.shape;s=state.shape[-1]
    out=torch.empty_like(gate)
    _scan[(batch,tr.cdiv(inner,4))](x,gate,dt,bias,a,d,state,slots,out,inner,length,s,head_dim,reset,
        4,tr.next_power_of_2(s),num_warps=4,enable_fp_fusion=False)
    return out
