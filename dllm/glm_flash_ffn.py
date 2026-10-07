"""Explicit Flash clipped FFN policies over shared GLM routing equations."""
import torch
from .glm_ffn import GlmFeedForward
from .glm_packed_ffn import PackedGlmFeedForward


def _limit(value):
    if type(value) not in (int,float) or not 0<value<=torch.finfo(torch.float32).max:
        raise ValueError('Flash requires a finite positive activation limit')
    return value


class FlashFeedForward(GlmFeedForward):
    def __init__(self,weights,*,activation_limit=10.,**options):
        super().__init__(weights,activation_limit=_limit(activation_limit),**options)


class PackedFlashFeedForward(PackedGlmFeedForward):
    def __init__(self,experts,*,activation_limit=10.,**options):
        super().__init__(experts,activation_limit=_limit(activation_limit),**options)
