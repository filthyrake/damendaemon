"""Bounded equal-prompt-length CPU Flash batches; no serving admission change."""
from .glm_generation import TokenStream
from .glm_flash_generation import FlashReferenceGenerator
from .reference_batch import prepare_batch, generate_prepared_batch


class FlashBatchReferenceGenerator(FlashReferenceGenerator):
    def generate_batch(self, conversations, *, max_new_tokens=128,
                       reasoning_effort=None, clear_thinking=False):
        prompts, limits = prepare_batch(self.text, conversations, max_new_tokens, self.max_context,
                                        reasoning_effort=reasoning_effort, clear_thinking=clear_thinking)
        streams = [TokenStream(self.text) for _ in prompts]
        yield from generate_prepared_batch(self, prompts, limits, streams,
            mapped_vocab_size=self.text.tokenizer_size, prefill=self.model.prefill, decode=lambda *args, **kwargs: self.model.forward(*args, **kwargs))
