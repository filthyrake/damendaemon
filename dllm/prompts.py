"""Shared raw-prompt validation and chat/raw dispatch."""


def validate_lengths(executor,prompt_tokens,max_tokens):
    if getattr(executor,'is_encoder_decoder',False):
        if prompt_tokens>executor.context or max_tokens>executor.context:
            raise ValueError('encoder prompt or decoder output exceeds configured context')
    elif prompt_tokens+max_tokens>executor.context:
        raise ValueError('prompt plus max_tokens exceeds configured context')


def decoder_prefix(executor,prompt):
    """Repetition penalties apply to decoder history, not encoder input."""
    if getattr(executor,'is_encoder_decoder',False):
        return [executor.model.config['decoder_start_token_id']]
    return prompt


def validate_raw(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and all(type(token) is int and token >= 0 for token in prompt):
        return list(prompt)
    raise ValueError('prompt must be text or a nonempty list of token IDs')


def encode(model, messages, raw_prompt=None, enable_thinking=None, tools=None):
    if getattr(model, "task", "generation") != "generation":
        raise ValueError("this model supports embeddings, not generation")
    if raw_prompt is not None:
        if messages is not None or enable_thinking is not None or tools is not None:
            raise ValueError('raw prompts cannot include chat messages or template options')
        return model.raw_tokens(raw_prompt)
    options={'enable_thinking':enable_thinking} if enable_thinking is not None else {}
    if tools is not None:options['tools']=tools
    return model.chat_tokens(messages,**options)
