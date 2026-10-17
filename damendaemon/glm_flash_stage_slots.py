"""Independent Flash request slots preserving all hyperconnection streams."""
import torch.nn.functional as F
from .cpu_stage_slots import CpuStageSlots
from .glm_flash_layer_stage import FlashLayerStage
from .glm_flash_generation import flash_text_config


class FlashStageSlots(CpuStageSlots):
    stage_type=FlashLayerStage

    def _configuration(self,stage):
        config=getattr(stage,'config',None)
        return {} if config is None else flash_text_config(config)

    def _cache_per_slot(self,stage,context):return stage._cache_per_slot(context)

    @property
    def _hidden_shape(self):return (self.stage.streams,self.stage.dim)

    def _embed(self,tokens):
        return F.embedding(tokens,self.stage.embedding).unsqueeze(-2).expand(-1,-1,self.stage.streams,-1)
