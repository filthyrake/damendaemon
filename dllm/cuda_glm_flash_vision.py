"""Explicit bounded CUDA FP32 Flash vision with math attention."""
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .glm_flash_vision import _FlashVision


class CudaFlashVision(_FlashVision):
    def __init__(self,config,weights,**options):
        if not isinstance(weights,dict) or not weights:raise ValueError('CUDA Flash vision requires prepared weights')
        first=next(iter(weights.values()))
        if not isinstance(first,torch.Tensor) or first.device.type!='cuda':
            raise ValueError('CUDA Flash vision requires CUDA FP32 weights')
        super().__init__(config,weights,device=first.device,**options)

    @torch.inference_mode()
    def forward(self,patches,grids):
        # Deliberate numerical reference path; no fused-attention speed claim.
        with sdpa_kernel(SDPBackend.MATH):
            return super().forward(patches,grids)
