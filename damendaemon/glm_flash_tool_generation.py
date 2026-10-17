"""Explicit Flash binding over the verified shared bounded tool lifecycle."""
from .glm_flash_generation import FlashReferenceGenerator
from .glm_flash_tool_text import FlashToolText
from .glm_flash_tool_output import FlashToolTokenStream
from .glm_tool_generation import ToolGenerationMixin


class FlashToolReferenceGenerator(ToolGenerationMixin,FlashReferenceGenerator):
    text_type=FlashToolText
    stream_type=FlashToolTokenStream
