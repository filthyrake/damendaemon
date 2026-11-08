"""GLM-5.3 binding for the separately verified schema-bound tool wire format.

Delegation shim — decision logic lives in native/glm_tool_output_asm.s.
"""

from .glm_tool_output_asm import (GlmToolTokenStream, TokenStream, GlmToolText,
    FlashToolOutput, OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE, CONTROLS,
    ByteLevelStream)
