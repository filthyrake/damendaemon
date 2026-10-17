"""Complete Flash CPU pipelines with mixed-cache and residual-stream ownership."""
from .layer_pipeline import CpuLayerPipeline,LayerPipelineState as FlashPipelineState
from .glm_flash_layer_stage import FlashLayerStage
from .glm_flash_model import _batch


class FlashLayerPipeline(CpuLayerPipeline):
    stage_type=FlashLayerStage

    def __init__(self,stages):
        super().__init__(stages)
        self.streams=self.stages[0].streams
        if any(stage.streams!=self.streams for stage in self.stages):
            raise ValueError('Flash pipeline residual-stream counts disagree')

    @property
    def boundary_width(self):return self.stages[0].dim*self.streams

    @property
    def config(self):
        import copy
        return copy.deepcopy(getattr(self.stages[0],'config',None))

    def _stage_batch(self,state):return _batch(state.layers[0].attention)

    def cache_bytes_per_row(self,capacity):return sum(stage._cache_per_slot(capacity) for stage in self.stages)

    def _embeddings(self,values,all_logits):
        self.stages[0]._embeddings(values,False)
        self._input_budget(values.shape[:2],all_logits)

    def forward_embeddings(self,values,state,*,position,all_logits=False):
        self._embeddings(values,all_logits)
        return self._forward(values,state,position=position,all_logits=all_logits,
                             first_forward=self.stages[0].forward_embeddings)

    def prefill_embeddings(self,values,*,capacity,max_cache_bytes=256*1024**2,all_logits=False):
        self._embeddings(values,all_logits)
        if type(capacity) is not int or values.shape[1]>capacity:raise ValueError('Flash pipeline embeddings exceed capacity')
        state=self.new_state(values.shape[0],capacity,max_cache_bytes=max_cache_bytes)
        try:return self.forward_embeddings(values,state,position=0,all_logits=all_logits),state
        except BaseException as error:
            self._release_failed(state,error);raise
