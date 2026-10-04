"""Experimental FP32 block-SSD prefill; independent of recurrent decode."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _ssd(X,G,DT,BIAS,A,D,STATE,SLOTS,Y,I:tl.constexpr,T:tl.constexpr,S:tl.constexpr,H:tl.constexpr,
         RESET:tl.constexpr,BT:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    channel=tl.arange(0,H);state=tl.arange(0,S);time=tl.arange(0,BT)
    slot=tl.load(SLOTS+row)
    address=STATE+(slot*I+head*H+channel[None,:])*S+state[:,None]
    value=tl.full((S,H),0,tl.float32) if RESET else tl.load(address)
    decay=tl.load(A+head);bias=tl.load(BIAS+head);skip=tl.load(D+head)
    for start in range(tr.cdiv(T,BT)):
        token=start*BT+time;valid=token<T
        step=tl.load(DT+(row*T+token)*(I//H)+head,valid,other=0)+bias
        step=tl.maximum(step,0)+libdevice.log1p(tl.exp(-tl.abs(step)))
        step=tl.where(valid,step,0)
        # Explicit tiny prefix reduction preserves a compatible layout across
        # the chained dot products on the supported Triton compiler.
        prefix=decay*tl.sum(tl.where(time[:,None]>=time[None,:],step[None,:],0),1)
        last=tl.sum(tl.where(time==BT-1,prefix,0),0)
        b=tl.load(X+(row*T+token[:,None])*(I+2*S)+I+state[None,:],valid[:,None],other=0)
        c=tl.load(X+(row*T+token[:,None])*(I+2*S)+I+S+state[None,:],valid[:,None],other=0)
        x=tl.load(X+(row*T+token[:,None])*(I+2*S)+head*H+channel[None,:],valid[:,None],other=0)
        weighted=x*step[:,None]
        lower=tl.where(time[:,None]>=time[None,:],tl.exp(tl.minimum(prefix[:,None]-prefix[None,:],0)),0)
        scores=tl.dot(c,tl.trans(b),input_precision='tf32x3')*lower
        out=tl.dot(scores,weighted,input_precision='tf32x3')
        out+=tl.dot(c,value,input_precision='tf32x3')*tl.exp(prefix)[:,None]
        value=tl.exp(last)*value+tl.dot(tl.trans(b),weighted*tl.exp(last-prefix)[:,None],input_precision='tf32x3')
        gate=tl.load(G+(row*T+token[:,None])*I+head*H+channel[None,:],valid[:,None],other=0)
        out=(out+skip*x)*(gate/(1+tl.exp(-gate)))
        tl.store(Y+(row*T+token[:,None])*I+head*H+channel[None,:],out,valid[:,None])
    tl.store(address,value)


def scan(x,gate,dt,bias,a,d,state,slots,head_dim,reset):
    batch,length,inner=gate.shape
    if length<128 or inner!=1536 or head_dim!=64 or state.shape[-1]!=128:
        from .mamba2_kernels import scan as recurrent
        return recurrent(x,gate,dt,bias,a,d,state,slots,head_dim,reset)
    out=torch.empty_like(gate)
    _ssd[(batch,inner//head_dim)](x,gate,dt,bias,a,d,state,slots,out,inner,length,128,64,reset,16,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out
