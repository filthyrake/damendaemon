"""Owned local CPU composition of explicitly prepared GLM layer stages."""
from .layer_pipeline import CpuLayerPipeline,LayerPipelineState as GlmPipelineState
from .glm_layer_stage import GlmLayerStage


class GlmLayerPipeline(CpuLayerPipeline):
    stage_type=GlmLayerStage

    def _stage_batch(self,state):return state.layers[0].attention.latents.shape[0]

    def cache_bytes_per_row(self,capacity):
        return 4*capacity*sum(block.attention.rank+block.attention.rope_dim
            +(block.attention.indexer.width if block.attention.indexer is not None else 0)
            for stage in self.stages for block in stage.blocks)
