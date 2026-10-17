"""Explicit prepared CUDA Flash mixed stack; no automatic checkpoint dispatch."""
from .glm_flash_block import _FlashBlock
from .glm_flash_model import _FlashModel
from .glm_flash_ffn import _limit
from .cuda_glm_ffn import CudaPackedGlmFeedForward
from .cuda_glm_flash_primitives import CudaFlashHyperConnection,_device
from .cuda_glm_flash_linear import CudaFlashLinearAttention
from .cuda_glm_flash_sparse import CudaFlashSparseAttention


class CudaPackedFlashFeedForward(CudaPackedGlmFeedForward):
    def __init__(self,experts,*,activation_limit=10.,**options):
        super().__init__(experts,activation_limit=_limit(activation_limit),**options)


class CudaFlashBlock(_FlashBlock):
    def __init__(self,attention,feed_forward,attention_hyper,ffn_hyper,input_norm,post_norm,**options):
        device=_device(input_norm)
        if not isinstance(feed_forward,CudaPackedFlashFeedForward) or feed_forward.device!=device:
            raise ValueError('CUDA Flash block requires same-device clipped Flash FFN')
        super().__init__(attention,feed_forward,attention_hyper,ffn_hyper,input_norm,post_norm,
                         device=device,attention_types=(CudaFlashLinearAttention,CudaFlashSparseAttention),
                         ffn_types=(CudaPackedFlashFeedForward,),hyper_type=CudaFlashHyperConnection,
                         sparse_type=CudaFlashSparseAttention,**options)


class CudaFlashModel(_FlashModel):
    def __init__(self,embedding,blocks,norm,head,**options):
        super().__init__(embedding,blocks,norm,head,device=_device(embedding),block_type=CudaFlashBlock,**options)
