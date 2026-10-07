"""Owned equal-prompt-length batches for the GLM sparse-attention CPU model."""
from .glm_generation import ReferenceGenerator,TokenStream
from .reference_batch import prepare_batch,generate_prepared_batch


class BatchReferenceGenerator(ReferenceGenerator):
    def generate_batch(self,conversations,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        prompts,limits=prepare_batch(self.text,conversations,max_new_tokens,self.max_context,
            reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        streams=[TokenStream(self.text) for _ in prompts]
        yield from generate_prepared_batch(self,prompts,limits,streams,
            mapped_vocab_size=self.text.tokenizer_size,prefill=self.model.prefill,
            decode=lambda *args,**kwargs:self.model.forward(*args,**kwargs))
