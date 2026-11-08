"""Incremental Flash ordinary channels plus atomic schema-bound tool calls.

Delegation shim — decision logic lives in native/glm_flash_tool_output_asm.s.
"""

from io import StringIO

from .glm_flash_tool_output_asm import (FlashToolOutput, FlashToolTokenStream,
    GlmOutput, STOPS, MAX_TEXT_BYTES, CONTROLS, TokenStream, ByteLevelStream,
    FlashTools, OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE, _text,
    FlashToolText)
