"""Explicit single-sequence prepared CUDA GLM text/tool generation."""
from .glm_generation import ReferenceGenerator
from .glm_tool_generation import ToolGenerationMixin
from .glm_tool_text import GlmToolText
from .glm_tool_output import GlmToolTokenStream
from .cuda_glm_model import CudaGlmModel


class CudaGlmGenerator(ReferenceGenerator):
    model_type=CudaGlmModel
    device_type='cuda'


class CudaGlmToolGenerator(CudaGlmGenerator):
    text_type=GlmToolText
    stream_type=GlmToolTokenStream
    # The CPU batch engine is deliberately not inherited by this binding.
    generate_tools=ToolGenerationMixin.generate_tools
