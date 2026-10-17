"""Explicit single-sequence prepared CUDA Flash native text/tool generation."""
from .glm_generation import ReferenceGenerator
from .glm_flash_generation import FlashReferenceGenerator
from .glm_flash_text import FlashText
from .glm_flash_tool_text import FlashToolText
from .glm_flash_tool_output import FlashToolTokenStream
from .glm_tool_generation import ToolGenerationMixin
from .cuda_glm_flash_model import CudaFlashModel


class CudaFlashGenerator(ReferenceGenerator):
    model_type=CudaFlashModel
    device_type='cuda'
    text_type=FlashText
    _configuration=FlashReferenceGenerator._configuration


class CudaFlashToolGenerator(CudaFlashGenerator):
    text_type=FlashToolText
    stream_type=FlashToolTokenStream
    generate_tools=ToolGenerationMixin.generate_tools
