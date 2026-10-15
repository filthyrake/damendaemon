"""Explicit CUDA FP32 Flash recurrence and residual-stream primitives."""
import torch
from .glm_flash_kda import _recurrent_attention
from .glm_flash_hyper import _FlashHyperConnection


def _device(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cuda':
        raise ValueError('explicit Flash CUDA primitive requires CUDA tensors')
    return value.device


def recurrent_attention(query,key,value,log_decay,beta,initial_state=None,**budgets):
    return _recurrent_attention(query,key,value,log_decay,beta,initial_state,device=_device(query),**budgets)


class CudaFlashHyperConnection(_FlashHyperConnection):
    def __init__(self,projection,base,scale,**options):
        super().__init__(projection,base,scale,device=_device(projection),**options)
