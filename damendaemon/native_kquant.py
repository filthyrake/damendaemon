"""Optional separately built CUDA K-quant decode, preserving original byte rows."""
import ctypes
from functools import lru_cache
import os
import torch


@lru_cache(maxsize=1)
def library(path):
    handle=ctypes.CDLL(path)
    version=handle.damendaemon_kquant_abi_version;version.argtypes=[];version.restype=ctypes.c_int
    if version()!=1:raise RuntimeError('unsupported damendaemon K-quant CUDA library ABI')
    function=handle.damendaemon_kquant_gemv
    function.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*6+[ctypes.c_void_p]
    function.restype=ctypes.c_int
    return handle,function


def linear_if_enabled(x,weight,layout,bias=None):
    path=os.environ.get('DAMENDAEMON_KQUANT_CUDA_LIBRARY')
    if not path:return None
    width,spec=layout;rows=x.shape[0];outputs=spec[-1][1]
    if x.dtype!=torch.float16 or not 1<=rows<=4 or any(kind not in (12,13,14) for _,_,kind,_ in spec):
        return None
    if (not x.is_cuda or not weight.is_cuda or x.device!=weight.device
            or not x.is_contiguous() or not weight.is_contiguous() or weight.dtype!=torch.uint8
            or x.ndim!=2 or x.shape[1]!=width or width%256):
        raise ValueError('invalid native K-quant buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype
            or tuple(bias.shape)!=(outputs,) or not bias.is_contiguous()):
        raise ValueError('invalid native K-quant bias')
    previous=0;byte_end=0
    for first,last,kind,offset in spec:
        if first!=previous or last<=first or offset!=byte_end:
            raise ValueError('invalid native K-quant segments')
        byte_end+=(last-first)*(width//256)*{12:144,13:176,14:210}[kind]
        previous=last
    if byte_end!=weight.numel():raise ValueError('invalid native K-quant byte count')
    _,function=library(path)
    output=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
    with torch.cuda.device(x.device):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        for first,last,kind,offset in spec:
            error=function(x.data_ptr(),weight.data_ptr()+offset,None if bias is None else bias.data_ptr(),
                output.data_ptr(),rows,outputs,width,first,last-first,kind,stream)
            if error:raise RuntimeError(f'native K-quant CUDA launch failed: {error}')
    return output
