"""Explicit bounded Flash image/image-tool generation over CPU layer pipelines."""
from .glm_flash_layer_pipeline import FlashLayerPipeline
from .glm_flash_image_generation import FlashImageReferenceGenerator
from .glm_flash_image_tool_generation import FlashImageToolReferenceGenerator


class FlashPipelineImageReferenceGenerator(FlashImageReferenceGenerator):
    model_type=FlashLayerPipeline


class FlashPipelineImageToolReferenceGenerator(FlashImageToolReferenceGenerator):
    model_type=FlashLayerPipeline
