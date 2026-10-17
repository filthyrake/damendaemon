"""Explicit prepared CUDA GLM composition; no automatic loading or serving."""
import torch
from .glm_block import _GlmBlock
from .glm_model import _GlmModel
from .cuda_glm_cached_attention import CudaCachedGlmAttention
from .cuda_glm_ffn import CudaPackedGlmFeedForward


class CudaGlmBlock(_GlmBlock):
    def __init__(self,attention,feed_forward,input_norm,post_norm,*,epsilon=1e-5):
        if not isinstance(attention,CudaCachedGlmAttention):
            raise ValueError('CUDA GLM block requires explicit CUDA attention')
        super().__init__(attention,feed_forward,input_norm,post_norm,epsilon=epsilon,device=attention.device,
                         attention_type=CudaCachedGlmAttention,ffn_type=CudaPackedGlmFeedForward)


class CudaGlmModel(_GlmModel):
    def __init__(self,embedding,blocks,norm,head,*,epsilon=1e-5,max_output_bytes=64*1024**2):
        if not isinstance(embedding,torch.Tensor) or embedding.device.type!='cuda':
            raise ValueError('CUDA GLM model requires a CUDA embedding')
        super().__init__(embedding,blocks,norm,head,epsilon=epsilon,max_output_bytes=max_output_bytes,
                         device=embedding.device,block_type=CudaGlmBlock)
