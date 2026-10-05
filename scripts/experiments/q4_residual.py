"""Isolated legacy-Q4 candidate; no normal runtime dispatch is changed."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch
from dllm.native_kquant_residual import library as base_library


@lru_cache(maxsize=None)
def library(path):
    handle, quantize, _ = base_library(path)
    multiply=handle.dllm_q4_residual_gemv
    multiply.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*3+[ctypes.c_void_p]
    multiply.restype=ctypes.c_int
    return handle,quantize,multiply


def activation_terms(path):
    handle,_,_=library(path)
    function=getattr(handle,'dllm_q4_activation_terms',None)
    if function is None:return 2  # Earlier two-term experimental builds.
    function.argtypes=[];function.restype=ctypes.c_int
    terms=function()
    if terms not in (1,2):raise RuntimeError('unsupported experimental activation quantizer')
    return terms


def linear(x,weights,scales,bias=None):
    rows,width=x.shape;outputs=weights.shape[0]
    if (not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous() or not 1<=rows<=4 or width%256
            or weights.device!=x.device or weights.dtype!=torch.uint8 or not weights.is_contiguous()
            or tuple(weights.shape)!=(outputs,width//2) or scales.device!=x.device
            or scales.dtype!=torch.float16 or not scales.is_contiguous() or tuple(scales.shape)!=(outputs,width//32)):
        raise ValueError('unsupported experimental Q4 buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or tuple(bias.shape)!=(outputs,)
            or not bias.is_contiguous()):raise ValueError('invalid experimental Q4 bias')
    _,quantize,multiply=library(os.environ['DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY'])
    a=torch.empty_like(x,dtype=torch.int8);b=torch.empty_like(a)
    xs=torch.empty((rows,width//32,4),dtype=torch.float32,device=x.device)
    output=torch.empty((rows,outputs),dtype=x.dtype,device=x.device)
    with torch.cuda.device(x.device):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        error=quantize(x.data_ptr(),a.data_ptr(),b.data_ptr(),xs.data_ptr(),rows,width,stream)
        if error:raise RuntimeError(f'experimental Q4 quantize failed: {error}')
        error=multiply(a.data_ptr(),b.data_ptr(),xs.data_ptr(),weights.data_ptr(),scales.data_ptr(),
                       bias.data_ptr() if bias is not None else None,output.data_ptr(),rows,outputs,width,stream)
        if error:raise RuntimeError(f'experimental Q4 multiply failed: {error}')
    return output


def install_validation_dispatch():
    from dllm import native_q4
    path=Path(os.environ['DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY']).resolve()
    library(str(path)); original=native_q4.linear_if_enabled
    def candidate(x,weights,scales,bias=None):
        if x.is_cuda and x.dtype==torch.float16 and x.shape[0]==1 and weights.shape[0]>=1024 and x.shape[1]>=4096 and x.shape[1]%256==0:
            return linear(x,weights,scales,bias)
        return original(x,weights,scales,bias)
    native_q4.linear_if_enabled=candidate
    terms=activation_terms(str(path))
    return dict(name=f'experimental Q4 {terms}-term int8',activation_terms=terms,
                activation_scaling='absmax/127' if terms==1 else 'two dyadic residual terms',
                library=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arithmetic='FP32 coefficient scaling; not rounded FP16 coefficients',
                scope='single row, width >=4096 divisible by 256, outputs >=1024',production_dispatch=False)
