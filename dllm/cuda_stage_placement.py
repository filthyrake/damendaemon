"""Explicit placement of only the tensors owned by a prepared CPU layer range."""
from dataclasses import dataclass
import torch
from .glm_layer_stage import GlmLayerStage
from .glm_flash_layer_stage import FlashLayerStage
from .deepseek_layer_stage import DeepSeekLayerStage
from . import cuda_glm_placement as glm
from . import cuda_glm_flash_placement as flash
from . import cuda_deepseek_placement as deepseek
from .cuda_packed_placement import _copy


@dataclass(frozen=True)
class PlacedCudaStage:
    stage: object
    raw_bytes: int
    device: torch.device


def _recipe(source,workspace_bytes):
    if isinstance(source,GlmLayerStage):module=glm;options={}
    elif isinstance(source,FlashLayerStage):module=flash;options={}
    elif isinstance(source,DeepSeekLayerStage):
        module=deepseek;options=dict(workspace_bytes=workspace_bytes,max_output_bytes=source.max_output_bytes)
    else:raise ValueError('CUDA range placement requires an explicit prepared CPU stage')
    return module,options


@torch.no_grad()
def place_cuda_stage(source,*,device,max_bytes,workspace_bytes=256*1024**2):
    """No surrounding model allocation, transfer policy or transport is implied.

    Existing range loaders determine owned weights. Revalidate the complete CPU
    composition before copying unique tensor objects; retain shared expert and
    packed-row identities. Raw bytes exclude allocator/cache/compute overhead.
    DeepSeek uses the explicit outer workspace bound; GLM/Flash retain their
    prepared per-component limits. Caller must keep source contents immutable.
    """
    device=torch.device(device)
    if device.type!='cuda' or device.index is None or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('CUDA stage placement requires indexed device and positive payload budget')
    module,options=_recipe(source,workspace_bytes)
    tensors=module._tensors(source,stage=True)
    needed=sum(v.numel()*v.element_size() for v in tensors.values())
    if needed>max_bytes:raise ValueError('CUDA stage raw payload budget exceeded')
    if module is deepseek:module._policy(source,**options,stage=True)
    module._build(source,tensors,cuda=False,stage=True,**options)
    copied={key:_copy(value,device) for key,value in tensors.items()}
    return PlacedCudaStage(module._build(source,copied,cuda=True,stage=True,**options),needed,device)
