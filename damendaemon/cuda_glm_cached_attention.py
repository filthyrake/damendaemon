"""Explicit CUDA GLM full-indexer and owned full/shared attention cache."""
import torch
from .glm_indexer import _GlmIndexer
from .glm_cached_attention import _CachedGlmAttention


def _device(weights,key):
    value=weights.get(key) if isinstance(weights,dict) else None
    device=getattr(value,'device',None)
    if not isinstance(device,torch.device) or device.type!='cuda':
        raise ValueError('explicit CUDA GLM weights require an indexed CUDA device')
    return device


class CudaGlmIndexer(_GlmIndexer):
    def __init__(self,weights,**options):
        super().__init__(weights,device=_device(weights,'wk.weight'),**options)


class CudaCachedGlmAttention(_CachedGlmAttention):
    def __init__(self,weights,**options):
        super().__init__(weights,device=_device(weights,'q_a_proj.weight'),**options)
