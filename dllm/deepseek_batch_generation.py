"""Bounded equal-length DeepSeek V4 CPU text batches; no serving admission."""
from .deepseek_generation import ReferenceGenerator, TokenStream
from .reference_batch import prepare_batch, generate_prepared_batch


class BatchReferenceGenerator(ReferenceGenerator):
    def generate_batch(self, conversations, *, max_new_tokens=128,
                       enable_thinking=False, drop_thinking=True, tools=None):
        prompts, limits = prepare_batch(self.text, conversations, max_new_tokens, self.max_context,
                                        enable_thinking=enable_thinking, drop_thinking=drop_thinking, tools=tools)
        names = frozenset(tool['function']['name'] for tool in tools) if tools is not None else frozenset()
        streams = [TokenStream(self.text, enable_thinking=enable_thinking,
                               allow_tool_calls=tools is not None) for _ in prompts]
        def finalize(row, events, reason):
            for event in events:
                if event.get('tool_calls'):
                    if any(call['function']['name'] not in names for call in event['tool_calls']):
                        raise ValueError('generated undeclared V4 batch tool')
                    reason = 'tool_calls'
            return reason
        def prefill(tokens, *, capacity, max_cache_bytes):
            return self.model.prefill(tokens, max_tokens=capacity, max_cache_bytes=max_cache_bytes)
        yield from generate_prepared_batch(self, prompts, limits, streams,
            mapped_vocab_size=self.text.vocab_size, prefill=prefill, decode=lambda *args, **kwargs: self.model.decode(*args, **kwargs), finalize=finalize)
