"""Explicit prepared CPU pipeline generation; ordinary model admission is unchanged."""
from .glm_layer_pipeline import GlmLayerPipeline
from .glm_batch_generation import BatchReferenceGenerator
from .glm_tool_generation import GlmToolReferenceGenerator


class PipelineReferenceGenerator(BatchReferenceGenerator):
    """Single/equal-length/ragged text generation over one complete pipeline."""
    model_type = GlmLayerPipeline


class PipelineToolReferenceGenerator(GlmToolReferenceGenerator):
    """Pinned GLM native-tool grammar; calls are returned, never executed."""
    model_type = GlmLayerPipeline
