"""Owned Flash CPU still-image/tool generation, with explicit separate admission."""
from .glm_flash_image_generation import FlashImageReferenceGenerator
from .glm_flash_image_tool_text import FlashImageToolText,_render_image_tool_chat
from .glm_flash_tools import FlashTools,_json,_dump
from .glm_flash_tool_output import FlashToolTokenStream


class FlashImageToolReferenceGenerator(FlashImageReferenceGenerator):
    text_type=FlashImageToolText

    def generate_image_tools(self,messages,images,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:
            raise ValueError('Flash image/tool output length must be 1..4096 tokens')
        declarations=_json(_dump(tools));codec=FlashTools(declarations)
        prompt,count=_render_image_tool_chat(messages,declarations,reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        stream=FlashToolTokenStream(self.text,codec)
        tokenize=lambda grids:self.text._expand_image_tokens(prompt,count,grids)
        try:yield from self._generate_images(images,count,tokenize,max_new_tokens,stream=stream)
        finally:stream.close()
