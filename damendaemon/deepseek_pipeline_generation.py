"""Explicit DeepSeek pipeline native single/batch text and tool generation."""
from .deepseek_batch_generation import BatchReferenceGenerator
from .deepseek_layer_pipeline import DeepSeekLayerPipeline


class DeepSeekPipelineReferenceGenerator(BatchReferenceGenerator):
    model_type=DeepSeekLayerPipeline
