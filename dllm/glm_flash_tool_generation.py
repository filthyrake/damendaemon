"""Owned single-request Flash CPU tool generation; calls are data, never executed."""
from .glm_flash_generation import FlashReferenceGenerator
from .glm_flash_tool_text import FlashToolText
from .glm_flash_tools import FlashTools, _json, _dump
from .glm_flash_tool_output import FlashToolTokenStream


class FlashToolReferenceGenerator(FlashReferenceGenerator):
    text_type=FlashToolText

    def generate_tools(self,messages,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:
            raise ValueError('Flash tool output length must be 1..4096 tokens')
        declarations=_json(_dump(tools));codec=FlashTools(declarations)
        tokens=self.text.tool_chat_tokens(messages,declarations,reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        stream=FlashToolTokenStream(self.text,codec)
        try:yield from self._generate_tokens(tokens,max_new_tokens,stream=stream)
        finally:stream.close()
