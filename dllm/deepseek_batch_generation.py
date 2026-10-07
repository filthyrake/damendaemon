"""Bounded equal-length DeepSeek V4 CPU text batches; no serving admission."""
from .deepseek_generation import ReferenceGenerator, TokenStream
from .reference_batch import prepare_batch, generate_prepared_batch


class BatchReferenceGenerator(ReferenceGenerator):
    def generate_batch(self, conversations, *, max_new_tokens=128,
                       enable_thinking=False, drop_thinking=True, tools=None):
        if tools is not None:
            raise ValueError('DeepSeek reference batches do not yet support tool calls')
        prompts, limits = prepare_batch(self.text, conversations, max_new_tokens, self.max_context,
                                        enable_thinking=enable_thinking, drop_thinking=drop_thinking)
        streams = [TokenStream(self.text, enable_thinking=enable_thinking) for _ in prompts]
        def prefill(tokens, *, capacity, max_cache_bytes):
            return self.model.prefill(tokens, max_tokens=capacity, max_cache_bytes=max_cache_bytes)
        yield from generate_prepared_batch(self, prompts, limits, streams,
            mapped_vocab_size=self.text.vocab_size, prefill=prefill, decode=lambda *args, **kwargs: self.model.decode(*args, **kwargs))
