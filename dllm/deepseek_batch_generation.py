"""Bounded equal-length DeepSeek V4 CPU text batches; no serving admission."""
from .deepseek_generation import ReferenceGenerator, TokenStream
from .deepseek_tool_prompt import prepare_tools, validate_tool_arguments
from .reference_batch import prepare_batch, generate_prepared_batch


class BatchReferenceGenerator(ReferenceGenerator):
    def generate_batch(self, conversations, *, max_new_tokens=128,
                       enable_thinking=False, drop_thinking=True, tools=None):
        prompts, limits = prepare_batch(self.text, conversations, max_new_tokens, self.max_context,
                                        enable_thinking=enable_thinking, drop_thinking=drop_thinking, tools=tools)
        contract = prepare_tools(tools)[1] if tools is not None else {}
        streams = [TokenStream(self.text, enable_thinking=enable_thinking,
                               allow_tool_calls=tools is not None) for _ in prompts]
        def finalize(row, events, reason):
            for event in events:
                if event.get('tool_calls'):
                    validate_tool_arguments(event['tool_calls'], contract)
                    reason = 'tool_calls'
            return reason
        def prefill(tokens, *, capacity, max_cache_bytes):
            return self.model.prefill(tokens, max_tokens=capacity, max_cache_bytes=max_cache_bytes)
        yield from generate_prepared_batch(self, prompts, limits, streams,
            mapped_vocab_size=self.text.vocab_size, prefill=prefill, decode=lambda *args, **kwargs: self.model.decode(*args, **kwargs), finalize=finalize)
