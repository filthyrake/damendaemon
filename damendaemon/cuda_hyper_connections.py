"""Explicit bounded CUDA FP32 DeepSeek residual-stream primitives."""
import torch
from . import hyper_connections as core


def _admit(hidden,operation,max_workspace_bytes,max_output_bytes):
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=4 or hidden.dtype!=torch.float32
            or hidden.device.type!='cuda' or not 1<=hidden.shape[0]<=128
            or not 1<=hidden.shape[1]<=2**20 or not 1<=hidden.shape[2]<=16
            or not 1<=hidden.shape[3]<=2**21):
        raise ValueError('CUDA residual mixing requires bounded FP32 residual streams')
    b,t,h,d=hidden.shape;rows=b*t
    if any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes)):
        raise ValueError('invalid CUDA residual mixing budget')
    workspace=64*rows*(h*d+h*(h+2)+h*h+d)
    output=4*rows*{'reduce':d+h+h*h,'restore':h*d,'head':d}[operation]
    if workspace>max_workspace_bytes or output>max_output_bytes:
        raise ValueError('CUDA residual mixing workspace/output budget exceeded')
    return b,t,h,d


def _parameter(value,shape,device):
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32
            or value.device!=device or tuple(value.shape)!=tuple(shape)):
        raise ValueError('CUDA residual mixing needs matching same-device FP32 parameters')


def _options(norm_epsilon,connection_epsilon,iterations=20):
    for value in (norm_epsilon,connection_epsilon):
        core._positive(value,'CUDA residual epsilon')
    if type(iterations) is not int or not 1<=iterations<=128:
        raise ValueError('CUDA residual iteration count must be1..128')


@torch.no_grad()
def reduce_streams(hidden,projection,scale,bias,*,norm_epsilon=1e-6,connection_epsilon=1e-6,iterations=20,
                   max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Budget before scanning; retain DeepSeek projection-then-normalization order."""
    _,_,h,d=_admit(hidden,'reduce',max_workspace_bytes,max_output_bytes)
    _parameter(projection,(h*(h+2),h*d),hidden.device)
    _parameter(scale,(3,),hidden.device);_parameter(bias,(h*(h+2),),hidden.device)
    _options(norm_epsilon,connection_epsilon,iterations)
    return core.reduce_streams(hidden,projection,scale,bias,norm_epsilon=norm_epsilon,
                               connection_epsilon=connection_epsilon,iterations=iterations)


@torch.no_grad()
def restore_streams(output,residual,post,matrix,*,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    b,t,h,d=_admit(residual,'restore',max_workspace_bytes,max_output_bytes)
    _parameter(output,(b,t,d),residual.device);_parameter(post,(b,t,h),residual.device)
    _parameter(matrix,(b,t,h,h),residual.device)
    return core.restore_streams(output,residual,post,matrix)


@torch.no_grad()
def reduce_head(hidden,projection,scale,bias,*,norm_epsilon=1e-6,connection_epsilon=1e-6,
                max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    _,_,h,d=_admit(hidden,'head',max_workspace_bytes,max_output_bytes)
    _parameter(projection,(h,h*d),hidden.device)
    _parameter(scale,(1,),hidden.device);_parameter(bias,(h,),hidden.device)
    _options(norm_epsilon,connection_epsilon)
    return core.reduce_head(hidden,projection,scale,bias,norm_epsilon=norm_epsilon,connection_epsilon=connection_epsilon)
