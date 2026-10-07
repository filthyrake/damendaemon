"""Bounded, functional CPU FP32 Kimi delta-attention recurrence for GLM Flash.

Equations independently transcribed from Transformers v5.17.0 glm5_next.
Per-key-dimension log decay acts before the delta update; query and key use
sqrt(sum(square)+1e-6) normalization. No convolution, model or GPU admission.
"""
import torch


def _tensor(value,shape):
    return (isinstance(value,torch.Tensor) and value.device.type=='cpu'
            and value.dtype==torch.float32 and tuple(value.shape)==tuple(shape))


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('nonfinite Flash KDA value')


@torch.no_grad()
def recurrent_attention(query,key,value,log_decay,beta,initial_state=None,*,
                        max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Return output [B,T,H,V] and independently owned final state [B,H,K,V].

    Inputs and initial state are never modified, including on failure. Limits
    cover conservative temporary tensors and both returned allocations, excluding
    borrowed inputs. This reference accepts finite log decay in [-100,0] and
    update gates in [0,1]; pinned Flash gates occupy the narrower [-5,0] interval.
    """
    if (not isinstance(query,torch.Tensor) or query.ndim!=4
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid Flash KDA geometry or budgets')
    b,t,h,k=query.shape
    if (not 1<=b<=65536 or not 1<=t<=2**20 or not 1<=h<=4096 or not 1<=k<=4096
            or not _tensor(query,(b,t,h,k)) or not _tensor(key,(b,t,h,k))
            or not _tensor(log_decay,(b,t,h,k)) or not _tensor(beta,(b,t,h))
            or not isinstance(value,torch.Tensor) or value.ndim!=4
            or not 1<=value.shape[-1]<=4096 or not _tensor(value,(b,t,h,value.shape[-1]))):
        raise ValueError('invalid Flash KDA CPU FP32 tensor geometry')
    v=value.shape[-1];state_shape=(b,h,k,v)
    if initial_state is not None and not _tensor(initial_state,state_shape):
        raise ValueError('invalid Flash KDA initial state')
    # Input validation masks plus per-token normalization, decay, prediction,
    # delta outer product and state copies. No sequence-squared allocation.
    workspace=16*b*t*h*(3*k+v+1)+64*b*h*(k*v+4*k+4*v)
    output=4*b*h*(t*v+k*v)
    if workspace>max_workspace_bytes or output>max_output_bytes:
        raise ValueError('Flash KDA workspace/output budget exceeded')
    _finite(query,key,value,log_decay,beta)
    if (log_decay>0).any() or (log_decay < -100).any() or (beta<0).any() or (beta>1).any():
        raise ValueError('invalid Flash KDA decay or update gate range')
    if initial_state is not None:_finite(initial_state)
    state=(torch.zeros(state_shape,dtype=torch.float32,device='cpu') if initial_state is None else initial_state.clone())
    result=torch.empty((b,t,h,v),dtype=torch.float32,device='cpu')
    for token in range(t):
        q=query[:,token];kk=key[:,token]
        qsum=q.square().sum(-1,keepdim=True);ksum=kk.square().sum(-1,keepdim=True)
        _finite(qsum,ksum)
        q=q/torch.sqrt(qsum+1e-6)/(k**.5);kk=kk/torch.sqrt(ksum+1e-6)
        state=state*log_decay[:,token].exp().unsqueeze(-1)
        prediction=(state*kk.unsqueeze(-1)).sum(-2)
        delta=(value[:,token]-prediction)*beta[:,token].unsqueeze(-1)
        state=state+kk.unsqueeze(-1)*delta.unsqueeze(-2)
        attended=(state*q.unsqueeze(-1)).sum(-2)
        _finite(state,attended)
        result[:,token]=attended
    return result,state
