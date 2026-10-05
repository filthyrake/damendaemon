"""Validation-only wrapper; no production runtime imports this experiment."""
import ctypes
from functools import lru_cache
import os
import torch


@lru_cache(maxsize=1)
def library(path):
    handle=ctypes.CDLL(path)
    abi=handle.dllm_residual_abi_version;abi.argtypes=[];abi.restype=ctypes.c_int
    if abi()!=1:raise RuntimeError('unsupported residual experiment ABI')
    quantize=handle.dllm_residual_quantize
    quantize.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    quantize.restype=ctypes.c_int
    multiply=handle.dllm_residual_gemv
    multiply.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6+[ctypes.c_void_p]
    multiply.restype=ctypes.c_int
    return handle,quantize,multiply


def linear(x,weight,layout,bias=None):
    width,spec=layout;rows=x.shape[0];outputs=spec[-1][1]
    if (x.ndim!=2 or not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous()
            or not 1<=rows<=4 or x.shape[1]!=width or width%256
            or weight.dtype!=torch.uint8 or weight.device!=x.device or not weight.is_contiguous()
            or any(kind not in (12,13,14) for _,_,kind,_ in spec)):
        raise ValueError('unsupported residual experiment buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or tuple(bias.shape)!=(outputs,)
            or not bias.is_contiguous()):raise ValueError('invalid experiment bias')
    previous=0;byte_end=0
    for first,last,kind,offset in spec:
        if first!=previous or last<=first or offset!=byte_end:raise ValueError('invalid experiment segments')
        byte_end+=(last-first)*(width//256)*{12:144,13:176,14:210}[kind];previous=last
    if byte_end!=weight.numel():raise ValueError('invalid experiment byte count')
    _,quantize,multiply=library(os.environ['DLLM_EXPERIMENTAL_RESIDUAL_LIBRARY'])
    first=torch.empty_like(x,dtype=torch.int8);second=torch.empty_like(first)
    scales=torch.empty((rows,width//32,4),device=x.device,dtype=torch.float32)
    output=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
    with torch.cuda.device(x.device):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        error=quantize(x.data_ptr(),first.data_ptr(),second.data_ptr(),scales.data_ptr(),rows,width,stream)
        if error:raise RuntimeError(f'residual quantization launch failed: {error}')
        for begin,end,kind,offset in spec:
            error=multiply(first.data_ptr(),second.data_ptr(),scales.data_ptr(),weight.data_ptr()+offset,
                None if bias is None else bias.data_ptr(),output.data_ptr(),rows,outputs,width,begin,end-begin,kind,stream)
            if error:raise RuntimeError(f'residual matrix launch failed: {error}')
    return output


def install_validation_dispatch():
    """Install only inside an explicitly selected validation process.

    Restrict to the measured winning A2 shapes. All other projections retain
    production dispatch; this is not a general device autotuning policy.
    """
    import hashlib
    from pathlib import Path
    from dllm import native_kquant

    path = Path(os.environ['DLLM_EXPERIMENTAL_RESIDUAL_LIBRARY']).resolve()
    library(str(path))  # Fail before running any model checks.
    original = native_kquant.linear_if_enabled

    def experimental(x, weight, layout, bias=None):
        width, spec = layout
        if (x.ndim == 2 and x.shape[0] == 1 and width == 2048
                and x.dtype == torch.float16 and x.is_cuda
                and all(kind in (12, 13, 14) for _, _, kind, _ in spec)):
            return linear(x, weight, layout, bias)
        return original(x, weight, layout, bias)

    native_kquant.linear_if_enabled = experimental
    return dict(name='residual-int8-v2', scope='FP16 single-row width-2048 K-quants',
                library=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_backend_label_excludes_experimental_override=True)
