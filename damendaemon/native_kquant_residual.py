"""Optional two-term int8 CUDA decode for a narrowly qualified K-quant scope."""
import ctypes
from functools import lru_cache
import os
import torch


@lru_cache(maxsize=1)
def library(path):
    handle=ctypes.CDLL(path)
    abi=handle.damendaemon_residual_abi_version;abi.argtypes=[];abi.restype=ctypes.c_int
    if abi()!=1:raise RuntimeError('unsupported residual CUDA ABI')
    quantize=handle.damendaemon_residual_quantize
    quantize.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    quantize.restype=ctypes.c_int
    multiply=handle.damendaemon_residual_gemv
    multiply.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6+[ctypes.c_void_p]
    multiply.restype=ctypes.c_int
    return handle,quantize,multiply


def linear(x,weight,layout,bias=None,*,library_path):
    width,spec=layout;rows=x.shape[0];outputs=spec[-1][1]
    if (x.ndim!=2 or not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous()
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



def qualified_scope(config, layouts, device, dtype, precision, context):
    """Q5 is explicitly excluded: its forced post-EOS numerical gate is open."""
    tinyllama = all(config.get(key)==value for key,value in dict(
        intermediate_size=5632,num_hidden_layers=22,num_attention_heads=32,
        num_key_value_heads=4,head_dim=64,vocab_size=32000,max_position_embeddings=2048).items())
    tinyllama = (tinyllama and config.get('rope_scaling') is None
                 and config.get('rope_theta',10000)==10000 and not config.get('sliding_window'))
    context_limit=2048 if tinyllama else 512
    return (device == 'cuda' and dtype == torch.float16 and precision == 'mixed' and 0 < context <= context_limit
            and config.get('model_type') == 'llama' and config.get('hidden_size') == 2048
            and bool(layouts)
            and {kind for _, spec in layouts.values() for _, _, kind, _ in spec} == {12, 14})


def select(config, layouts, device, dtype, precision, context):
    path = os.environ.get('DAMENDAEMON_KQUANT_RESIDUAL_LIBRARY')
    if not path or not qualified_scope(config, layouts, device, dtype, precision, context):
        return None
    library(path)

    def decode(x, weight, layout, bias=None):
        if x.ndim != 2 or x.shape[0] != 1 or layout[0] != 2048:
            return None
        return linear(x, weight, layout, bias, library_path=path)

    return decode
