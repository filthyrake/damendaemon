"""Shared bounded tool generation with explicit GLM sparse-attention binding."""
from .glm_generation import ReferenceGenerator
from .glm_tool_text import GlmToolText
from .glm_tool_output import GlmToolTokenStream
from .glm_flash_tools import FlashTools, _json, _dump
from .reference_batch import prepare_batch, generate_prepared_batch


class ToolGenerationMixin:
    """Use only with a family whose pinned prompt/wire contract is verified."""
    def generate_tools(self,messages,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:
            raise ValueError('GLM tool output length must be 1..4096 tokens')
        declarations=_json(_dump(tools));codec=FlashTools(declarations)
        tokens=self.text.tool_chat_tokens(messages,declarations,reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        stream=self.stream_type(self.text,codec)
        try:yield from self._generate_tokens(tokens,max_new_tokens,stream=stream)
        finally:stream.close()

    def generate_batch_tools(self,conversations,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False,ragged=False):
        declarations=_json(_dump(tools));codec=FlashTools(declarations)
        def tokenize(messages,**options):
            return self.text.tool_chat_tokens(messages,declarations,**options)
        prompts,limits=prepare_batch(self.text,conversations,max_new_tokens,self.max_context,tokenize=tokenize,
                                     reasoning_effort=reasoning_effort,clear_thinking=clear_thinking,ragged=ragged)
        streams=[self.stream_type(self.text,codec) for _ in prompts]
        def finalize(row,events,reason):
            return 'tool_calls' if any(event.get('tool_calls') for event in events) else reason
        try:
            yield from generate_prepared_batch(self,prompts,limits,streams,mapped_vocab_size=self.text.tokenizer_size,
                prefill=self.model.prefill,decode=lambda *args,**kwargs:self.model.forward(*args,**kwargs),finalize=finalize)
        finally:
            for stream in streams:stream.close()


class GlmToolReferenceGenerator(ToolGenerationMixin,ReferenceGenerator):
    text_type=GlmToolText
    stream_type=GlmToolTokenStream
