"""Optional separately built CUDA Q4 decode kernel; explicit startup opt-in."""
import ctypes
from functools import lru_cache
import os
import torch


@lru_cache(maxsize=1)
def library(path):
    handle=ctypes.CDLL(path)
    version=handle.damendaemon_q4_abi_version
    version.argtypes=[]
    version.restype=ctypes.c_int
    if version()!=1:
        raise RuntimeError('unsupported damendaemon Q4 CUDA library ABI')
    function=handle.damendaemon_q4_gemv
    function.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
    function.restype=ctypes.c_int
    return handle,function


def linear_if_enabled(x,weight,scales,bias=None):
    path=os.environ.get('DAMENDAEMON_Q4_CUDA_LIBRARY')
    if not path:
        return None
    rows,k=x.shape;n=weight.shape[0]
    # Only select the shape range where measurements beat the Tensor Core path.
    if rows!=1 or k<4096 or n<1024 or x.dtype!=torch.float16:
        return None
    if (not x.is_cuda or not weight.is_cuda or not scales.is_cuda
            or x.device!=weight.device or x.device!=scales.device
            or not x.is_contiguous() or not weight.is_contiguous() or not scales.is_contiguous()
            or weight.dtype!=torch.uint8 or scales.dtype!=torch.float16
            or k%32 or tuple(weight.shape)!=(n,k//2) or tuple(scales.shape)!=(n,k//32)):
        raise ValueError('invalid native Q4 buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype
                            or tuple(bias.shape)!=(n,) or not bias.is_contiguous()):
        raise ValueError('invalid native Q4 bias')
    _,function=library(path)
    output=torch.empty((rows,n),device=x.device,dtype=x.dtype)
    error=function(x.data_ptr(),weight.data_ptr(),scales.data_ptr(),
        None if bias is None else bias.data_ptr(),output.data_ptr(),rows,n,k,
        torch.cuda.current_stream(x.device).cuda_stream)
    if error:
        raise RuntimeError(f'native Q4 CUDA launch failed: {error}')
    return output
