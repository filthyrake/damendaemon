"""Independent DeepSeek CPU request slots carrying original routing token IDs."""
import torch
import torch.nn.functional as F
from .cpu_stage_slots import CpuStageSlots
from .deepseek_layer_stage import DeepSeekLayerStage


class DeepSeekStageSlots(CpuStageSlots):
    stage_type=DeepSeekLayerStage

    def _configuration(self,stage):return getattr(stage,'config',{})

    def _cache_per_slot(self,stage,context):return stage.cache_bytes_per_slot(context)

    @property
    def _hidden_shape(self):return (self.stage.streams,self.stage.dim)

    @property
    def _vocab_size(self):return self.stage.vocab_size

    def _embedding_tokens(self,tokens):self.stage._input(tokens,None,False)

    def _embed(self,tokens):
        return F.embedding(tokens,self.stage.embedding).unsqueeze(2).expand(-1,-1,self.stage.streams,-1)

    def _routing_tokens(self,hidden,tokens):self.stage._input(tokens,hidden,True)

    def _allocate_slots(self,ids):
        # DeepSeek creates its compressed state only from token-bearing prefill.
        self.stage.cleanup()

    def _usable(self):
        super()._usable()
        for state in self.states:
            if state is not None:state._usable()

    def _forward_slot(self,hidden,slot,start,tokens):
        if self.states[slot] is None:
            result,state=self.stage.prefill_activations(hidden,tokens,max_tokens=self.context,max_cache_bytes=self.max_cache_bytes)
            self.states[slot]=state
            return result
        # A continuation chunk uses the existing compressed caches. Preserve every
        # intermediate stream for the next stage, including compression boundaries.
        values=[]
        for offset in range(hidden.shape[1]):
            values.append(self.stage.decode_activations(hidden[:,offset:offset+1],tokens[:,offset:offset+1],
                self.states[slot],position=start+offset))
        return torch.cat(values,dim=1)

    def _project(self,hidden):return self.stage._logits(hidden)
