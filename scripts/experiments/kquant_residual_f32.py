"""Validation-only FP32 residual-int8 wrapper; separate ABI prevents accidental production selection."""
import ctypes
from functools import lru_cache
import os
import torch


@lru_cache(maxsize=1)
def library(path):
    handle=ctypes.CDLL(path)
    abi=handle.dllm_residual_abi_version;abi.argtypes=[];abi.restype=ctypes.c_int
    if abi()!=101:raise RuntimeError('unsupported residual CUDA ABI')
    quantize=handle.dllm_residual_quantize
    quantize.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    quantize.restype=ctypes.c_int
    multiply=handle.dllm_residual_gemv
    multiply.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6+[ctypes.c_void_p]
    multiply.restype=ctypes.c_int
    return handle,quantize,multiply


def _linear(x,weight,layout,bias=None,*,library_path):
    width,spec=layout;rows=x.shape[0];outputs=spec[-1][1]
    if (x.ndim!=2 or not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous()
            or not 1<=rows<=4 or x.shape[1]!=width or width%256
            or weight.dtype!=torch.uint8 or weight.device!=x.device or not weight.is_contiguous()
            or any(kind not in (12,13,14) for _,_,kind,_ in spec)):
        raise ValueError('unsupported residual CUDA buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or tuple(bias.shape)!=(outputs,)
            or not bias.is_contiguous()):raise ValueError('invalid residual bias')
    previous=0;byte_end=0
    for first,last,kind,offset in spec:
        if first!=previous or last<=first or offset!=byte_end:raise ValueError('invalid residual segments')
        byte_end+=(last-first)*(width//256)*{12:144,13:176,14:210}[kind];previous=last
    if byte_end!=weight.numel():raise ValueError('invalid residual byte count')
    _,quantize,multiply=library(library_path)
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



def linear(x, weight, layout, bias=None):
    return _linear(x, weight, layout, bias,
                   library_path=os.environ['DLLM_EXPERIMENTAL_RESIDUAL_F32_LIBRARY'])


def install_validation_dispatch():
    import hashlib
    from pathlib import Path
    from dllm import native_kquant

    path = Path(os.environ['DLLM_EXPERIMENTAL_RESIDUAL_F32_LIBRARY']).resolve()
    library(str(path))
    original = native_kquant.linear_if_enabled

    def experimental(x, weight, layout, bias=None):
        width, spec = layout
        if (x.ndim == 2 and 1 <= x.shape[0] <= 4 and width in (2048, 8192)
                and x.dtype == torch.float32 and x.is_cuda
                and all(kind in (12, 13, 14) for _, _, kind, _ in spec)):
            return linear(x, weight, layout, bias)
        return original(x, weight, layout, bias)

    native_kquant.linear_if_enabled = experimental
    return dict(name='residual-int8-float32', abi=101,
                scope='FP32 rows 1-4; widths 2048/8192; K-quant projections',
                library=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_backend_label_excludes_experimental_override=True)
