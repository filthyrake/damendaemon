"""Optional single-row integer-dot Q4 decode for a qualified Mistral GGUF scope."""
import ctypes
from functools import lru_cache
import os

import torch


@lru_cache(maxsize=None)
def library(path):
    # Keep loaded handles alive for closures and captured CUDA graphs.
    handle=ctypes.CDLL(path)
    abi=handle.dllm_q4_residual_abi_version;abi.argtypes=[];abi.restype=ctypes.c_int
    if abi()!=201:raise RuntimeError('unsupported Q4 residual CUDA ABI')
    quantize=handle.dllm_q4_residual_quantize
    quantize.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    quantize.restype=ctypes.c_int
    multiply=handle.dllm_q4_residual_gemv
    multiply.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*3+[ctypes.c_void_p]
    multiply.restype=ctypes.c_int
    return handle,quantize,multiply


def linear(x,weights,scales,bias=None,*,library_path):
    if x.ndim!=2 or weights.ndim!=2:raise ValueError('unsupported Q4 residual rank')
    rows,width=x.shape;outputs=weights.shape[0]
    if (not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous() or rows!=1 or width<=0 or width%256
            or outputs<=0 or weights.device!=x.device or weights.dtype!=torch.uint8 or not weights.is_contiguous()
            or tuple(weights.shape)!=(outputs,width//2) or scales.device!=x.device
            or scales.dtype!=torch.float16 or not scales.is_contiguous() or tuple(scales.shape)!=(outputs,width//32)):
        raise ValueError('unsupported Q4 residual buffers')
    if bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or tuple(bias.shape)!=(outputs,)
            or not bias.is_contiguous()):raise ValueError('invalid Q4 residual bias')
    _,quantize,multiply=library(library_path)
    first=torch.empty_like(x,dtype=torch.int8);second=torch.empty_like(first)
    activation_scales=torch.empty((rows,width//32,4),dtype=torch.float32,device=x.device)
    output=torch.empty((rows,outputs),dtype=x.dtype,device=x.device)
    with torch.cuda.device(x.device):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        error=quantize(x.data_ptr(),first.data_ptr(),second.data_ptr(),activation_scales.data_ptr(),rows,width,stream)
        if error:raise RuntimeError(f'Q4 residual quantize failed: {error}')
        error=multiply(first.data_ptr(),second.data_ptr(),activation_scales.data_ptr(),weights.data_ptr(),scales.data_ptr(),
                       bias.data_ptr() if bias is not None else None,output.data_ptr(),rows,outputs,width,stream)
        if error:raise RuntimeError(f'Q4 residual multiply failed: {error}')
    return output


def qualified_scope(config,quantization,device,dtype,precision,context):
    return (device=='cuda' and dtype==torch.float16 and precision=='mixed' and context<=512
            and quantization=='q4_0' and config.get('model_type')=='llama'
            and all(config.get(key)==value for key,value in dict(hidden_size=4096,intermediate_size=14336,
                num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,vocab_size=32000).items())
            and config.get('head_dim',128)==128)


def select(config,quantization,device,dtype,precision,context):
    path=os.environ.get('DLLM_Q4_RESIDUAL_LIBRARY')
    if not path or not qualified_scope(config,quantization,device,dtype,precision,context):return None
    library(path)
    def decode(x,weights,scales,bias=None):
        if x.ndim!=2 or x.shape[0]!=1 or x.shape[1]<4096 or x.shape[1]%256 or weights.shape[0]<1024:return None
        return linear(x,weights,scales,bias,library_path=path)
    return decode
