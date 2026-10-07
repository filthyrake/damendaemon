"""Explicit native Flash text/tool generation over complete CPU layer pipelines."""
from .glm_flash_layer_pipeline import FlashLayerPipeline
from .glm_flash_batch_generation import FlashBatchReferenceGenerator
from .glm_flash_tool_generation import FlashToolReferenceGenerator


class FlashPipelineReferenceGenerator(FlashBatchReferenceGenerator):
    model_type=FlashLayerPipeline


class FlashPipelineToolReferenceGenerator(FlashToolReferenceGenerator):
    model_type=FlashLayerPipeline
