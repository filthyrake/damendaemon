"""Explicit CUDA FP32 compressor composition with preallocation admission."""
from functools import partial
import torch
from .floating_compressor import _compress_hidden
from .cuda_compressed_pool import pool_projected,_state


@torch.no_grad()
def compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,*,state=None,
                    epsilon=1e-6,cache_transform=None,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or hidden.dtype!=torch.float32 or hidden.device.type!='cuda'
            or not 1<=hidden.shape[0]<=128 or not 0<=hidden.shape[1]<=2**20 or not 1<=hidden.shape[2]<=2**21
            or not isinstance(offsets,torch.Tensor) or offsets.ndim!=2 or not 1<=offsets.shape[0]<=128 or not 1<=offsets.shape[1]<=4096):
        raise ValueError('CUDA compressor requires bounded FP32 hidden/offset geometry')
    b,t,d=hidden.shape;r,c=offsets.shape
    if r==4 and c%2:raise ValueError('overlapping CUDA compression needs paired channels')
    width=c//2 if r==4 else c
    for value,shape in ((value_weight,(c,d)),(gate_weight,(c,d)),(offsets,(r,c)),(norm_weight,(width,))):
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.float32 or value.device!=hidden.device:
            raise ValueError('CUDA compressor needs matching same-device FP32 parameters')
    if (not isinstance(inverse_frequencies,torch.Tensor) or inverse_frequencies.ndim!=1
            or inverse_frequencies.dtype!=torch.float32 or inverse_frequencies.device!=hidden.device
            or not 1<=inverse_frequencies.numel()<=width//2):
        raise ValueError('invalid CUDA compressor frequencies')
    if state is not None:_state(state,b,c,r,hidden.device)
    position=0 if state is None else state.position
    if position+t>2**20:raise ValueError('CUDA compressor logical position exceeded')
    if any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes)):
        raise ValueError('invalid CUDA compressor budgets')
    count=(position+t)//r-position//r;buffer=b*r*(2 if r==4 else 1)*c
    output=4*(b*count*width+2*buffer)+8*count
    # Conservative projection, pooling, normalization, rotary/quantization and
    # validation allowance, including borrowed-parameter validation masks.
    workspace=256*(b*t*(d+2*c)+4*buffer+b*count*width+2*c*d+r*c+width)
    if workspace>max_workspace_bytes or output>max_output_bytes:
        raise ValueError('CUDA compressor workspace/output budget exceeded')
    pool=partial(pool_projected,max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
    return _compress_hidden(hidden,value_weight,gate_weight,offsets,norm_weight,inverse_frequencies,state=state,
                            epsilon=epsilon,cache_transform=cache_transform,pool_fn=pool,validate_finite=True)
