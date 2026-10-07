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
