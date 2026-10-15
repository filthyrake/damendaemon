"""Explicit same-device CUDA GLM/Flash ranges, without transport or placement."""
import torch
from .glm_layer_stage import _GlmLayerStage
from .glm_flash_layer_stage import _FlashLayerStage
from .cuda_glm_model import CudaGlmBlock
from .cuda_glm_flash_model import CudaFlashBlock


def _device(blocks,kind):
    if (not isinstance(blocks,(list,tuple)) or not blocks or any(not isinstance(b,kind) for b in blocks)
            or blocks[0].device.type!='cuda'):
        raise ValueError('CUDA stage requires explicitly prepared CUDA blocks')
    return blocks[0].device


class _CudaStageAdmission:
    def _admit_boundary(self,inputs,all_logits,*,embedded=False,activations=False):
        rank=4 if isinstance(self,_FlashLayerStage) else 3
        if self.first and not activations:rank=3 if embedded else 2
        if (not isinstance(inputs,torch.Tensor) or inputs.ndim!=rank or inputs.device!=self.device
                or not 1<=inputs.shape[0]<=128 or not 1<=inputs.shape[1]<=2**20 or type(all_logits) is not bool):
            raise ValueError('CUDA stage requires bounded same-device input chunks')
        streams=getattr(self,'streams',1);batch,length=inputs.shape[:2]
        required=4*batch*length*streams*self.dim
        if self.last and not activations:required=max(required,4*batch*(length if all_logits else 1)*self.head.shape[0])
        if required>self.max_output_bytes:raise ValueError('CUDA stage boundary output budget exceeded')

    def _tokens(self,inputs,all_logits):
        self._admit_boundary(inputs,all_logits)
        return super()._tokens(inputs,all_logits)

    def forward_activations(self,hidden,state,*,position):
        self._admit_boundary(hidden,False,activations=True)
        return super().forward_activations(hidden,state,position=position)


class CudaGlmLayerStage(_CudaStageAdmission,_GlmLayerStage):
    def __init__(self,blocks,**options):
        super().__init__(blocks,device=_device(blocks,CudaGlmBlock),block_type=CudaGlmBlock,**options)


class CudaFlashLayerStage(_CudaStageAdmission,_FlashLayerStage):
    def __init__(self,blocks,**options):
        super().__init__(blocks,device=_device(blocks,CudaFlashBlock),block_type=CudaFlashBlock,**options)

    def _embeddings(self,values,all_logits):
        self._admit_boundary(values,all_logits,embedded=True)
        return super()._embeddings(values,all_logits)
