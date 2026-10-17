"""Explicit bounded CUDA FP32 DeepSeek projected pooling; no model admission."""
import torch
from .compressed_pool import PoolState,pool_projected as _pool


def _budget(workspace,output,max_workspace_bytes,max_output_bytes):
    if any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes)):
        raise ValueError('invalid CUDA pooling budget')
    if workspace>max_workspace_bytes or output>max_output_bytes:
        raise ValueError('CUDA pooling workspace/output budget exceeded')


def _state(state,batch,channels,ratio,device):
    shape=(batch,ratio*(2 if ratio==4 else 1),channels)
    if (not isinstance(state,PoolState) or type(state.position) is not int or not 0<=state.position<=2**20
            or type(state.ratio) is not int or state.ratio!=ratio or (ratio==4 and channels%2)
            or any(not isinstance(v,torch.Tensor) or v.shape!=shape or v.dtype!=torch.float32 or v.device!=device
                   for v in (state.values,state.scores))):
        raise ValueError('invalid CUDA pooling state')


def _finite_state(state):
    if state is not None and (not torch.isfinite(state.values).all()
            or torch.isnan(state.scores).any() or torch.isposinf(state.scores).any()):
        raise ValueError('nonfinite CUDA pooling state')


@torch.no_grad()
def pool_projected(values,gates,offsets,*,state=None,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Admit geometry and conservative live storage before scanning or allocating.

    Inputs are borrowed and immutable. Output budget includes independent returned
    state and block starts. Workspace includes validation and overlap temporaries;
    allocator overhead and borrowed input residency are separate caller bounds.
    """
    if (not isinstance(values,torch.Tensor) or values.ndim!=3 or values.dtype!=torch.float32
            or values.device.type!='cuda' or not 1<=values.shape[0]<=128
            or not 0<=values.shape[1]<=2**20 or not 1<=values.shape[2]<=4096
            or not isinstance(gates,torch.Tensor) or gates.shape!=values.shape or gates.dtype!=values.dtype or gates.device!=values.device
            or not isinstance(offsets,torch.Tensor) or offsets.ndim!=2 or offsets.dtype!=values.dtype or offsets.device!=values.device
            or not 1<=offsets.shape[0]<=128 or offsets.shape[1]!=values.shape[2]):
        raise ValueError('CUDA pooling requires bounded matching same-device FP32 tensors')
    b,t,c=values.shape;r=offsets.shape[0]
    if r==4 and c%2:raise ValueError('overlapping CUDA pooling needs equal channel halves')
    if state is not None:_state(state,b,c,r,values.device)
    position=0 if state is None else state.position
    if position+t>2**20:raise ValueError('CUDA pooling logical position exceeded')
    width=c//2 if r==4 else c;buffer=b*r*(2 if r==4 else 1)*c
    count=(position+t)//r-position//r
    output=4*(b*count*width+2*buffer)+8*count
    workspace=64*(b*t*c+4*buffer+r*c)
    _budget(workspace,output,max_workspace_bytes,max_output_bytes)
    if any(not torch.isfinite(v).all() for v in (values,gates,offsets)):
        raise ValueError('nonfinite CUDA pooling input')
    _finite_state(state)
    result,replacement,starts=_pool(values,gates,offsets,state=state,validate_scores=True)
    if not torch.isfinite(result).all():raise ValueError('nonfinite CUDA pooling result')
    _finite_state(replacement)
    return result,replacement,starts


@torch.no_grad()
def reorder_pool(state,indices,*,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Return independent reordered slots after bounded same-device admission."""
    if (not isinstance(state,PoolState) or not isinstance(state.values,torch.Tensor)
            or state.values.ndim!=3 or state.values.device.type!='cuda'
            or not 1<=state.values.shape[0]<=128 or not 1<=state.values.shape[2]<=4096
            or type(state.ratio) is not int or not 1<=state.ratio<=128):
        raise ValueError('invalid CUDA pooling state')
    _state(state,state.values.shape[0],state.values.shape[2],state.ratio,state.values.device)
    if (not isinstance(indices,torch.Tensor) or indices.ndim!=1 or indices.dtype!=torch.int64
            or indices.device!=state.values.device or not 1<=indices.numel()<=128):
        raise ValueError('CUDA pool reorder needs bounded same-device int64 slots')
    output=8*indices.numel()*state.values.shape[1]*state.values.shape[2]
    _budget(2*output+8*state.values.numel(),output,max_workspace_bytes,max_output_bytes)
    if (indices<0).any() or (indices>=state.values.shape[0]).any():raise ValueError('invalid CUDA pooling slot index')
    _finite_state(state)
    return state.reorder(indices)
