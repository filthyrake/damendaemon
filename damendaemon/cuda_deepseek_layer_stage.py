"""Explicit bounded CUDA DeepSeek range carrying hash tokens and residual streams."""
import torch
from .deepseek_layer_stage import _DeepSeekLayerStage
from .cuda_deepseek_block import CudaDeepSeekBlock
from .cuda_deepseek_model import CudaDeepSeekModel


class CudaDeepSeekLayerStage(_DeepSeekLayerStage):
    _logits=CudaDeepSeekModel._logits

    def __init__(self,blocks,*,workspace_bytes=256*1024**2,**options):
        if (not isinstance(blocks,(list,tuple)) or not blocks or any(not isinstance(b,CudaDeepSeekBlock) for b in blocks)
                or blocks[0].device.type!='cuda' or type(workspace_bytes) is not int or not 2<=workspace_bytes<=2**31
                or any(b.backend.workspace_bytes>workspace_bytes//2 for b in blocks)):
            raise ValueError('CUDA DeepSeek stage requires explicit CUDA blocks and bounded scratch')
        self.workspace_bytes=workspace_bytes
        super().__init__(blocks,device=blocks[0].device,block_type=CudaDeepSeekBlock,**options)

    def _input(self,tokens,hidden,activations):
        if (not isinstance(tokens,torch.Tensor) or tokens.ndim!=2 or tokens.device!=self.device
                or not 1<=tokens.shape[0]<=128 or not 1<=tokens.shape[1]<=2**20):
            raise ValueError('CUDA DeepSeek stage requires bounded same-device token rows')
        b,t=tokens.shape
        outer=32*b*t*(self.streams*self.dim+self.dim+1)
        if self.last:outer+=8*sum(v.numel() for v in self.head.values())
        output=4*b*t*self.streams*self.dim
        if self.last and not activations:output=max(output,4*b*self.vocab_size)
        if outer>self.workspace_bytes//2 or output>self.max_output_bytes:
            raise ValueError('CUDA DeepSeek stage scratch/output budget exceeded')
        return super()._input(tokens,hidden,activations)
