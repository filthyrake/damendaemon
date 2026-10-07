"""Explicit GLM stage slots using shared per-request ownership."""
import torch.nn.functional as F
from .cpu_stage_slots import CpuStageSlots
from .glm_layer_stage import GlmLayerStage


class GlmStageSlots(CpuStageSlots):
    stage_type=GlmLayerStage

    def _configuration(self,stage):return getattr(stage,'config',{})

    def _cache_per_slot(self,stage,context):
        width=sum(block.attention.rank+block.attention.rope_dim
            +(block.attention.indexer.width if block.attention.indexer is not None else 0) for block in stage.blocks)
        return 4*context*width

    @property
    def _hidden_shape(self):return (self.stage.dim,)

    def _embed(self,tokens):return F.embedding(tokens,self.stage.embedding)
