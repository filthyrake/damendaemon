"""Joint Flash language/vision payload admission before the first CUDA copy."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from .glm_flash_vision import FlashVision
from .cuda_glm_flash_vision import CudaFlashVision
from .cuda_glm_flash_placement import _tensors,_build
from .cuda_packed_placement import _copy


@dataclass(frozen=True)
class PlacedCudaFlashImageModel:
    model: object
    vision: CudaFlashVision
    raw_bytes: int
    device: torch.device


def _vision(source,copied,*,cuda,max_weight_bytes):
    kind=CudaFlashVision if cuda else FlashVision
    result=kind(source.c,{k:copied[id(v)] for k,v in source.w.items()},max_weight_bytes=max_weight_bytes,
        max_tokens=source.max_tokens,max_workspace_bytes=source.max_workspace_bytes,max_output_bytes=source.max_output_bytes)
    if hasattr(source,'source_config'):
        if not isinstance(source.source_config,dict):raise ValueError('Flash visual source config must be a mapping')
        result.source_config=deepcopy(source.source_config)
    return result


@torch.no_grad()
def place_cuda_flash_image(source,vision,*,device,max_bytes):
    device=torch.device(device)
    if device.type!='cuda' or device.index is None or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('Flash image placement requires indexed CUDA device and payload budget')
    tensors=_tensors(source)
    if not isinstance(vision,FlashVision) or vision.c['out_hidden_size']!=source.embedding.shape[1]:
        raise ValueError('Flash image placement requires matching CPU vision/language geometry')
    for value in vision.w.values():
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu':raise ValueError('Flash vision source must be CPU')
        tensors[id(value)]=value
    if (hasattr(source,'config') and hasattr(vision,'source_config') and source.config!=vision.source_config):
        raise ValueError('Flash vision and language configurations disagree')
    needed=sum(v.numel()*v.element_size() for v in tensors.values())
    if needed>max_bytes:raise ValueError('Flash image aggregate raw payload budget exceeded')
    _build(source,tensors,cuda=False);_vision(vision,tensors,cuda=False,max_weight_bytes=max_bytes)
    copied={key:_copy(value,device) for key,value in tensors.items()}
    return PlacedCudaFlashImageModel(_build(source,copied,cuda=True),
        _vision(vision,copied,cuda=True,max_weight_bytes=max_bytes),needed,device)
