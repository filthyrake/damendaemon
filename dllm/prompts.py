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
        try:
            prompt.encode('utf-8')
        except UnicodeError as error:
            raise ValueError('prompt must be valid UTF-8') from error
        return prompt
    if isinstance(prompt, list) and prompt and all(type(token) is int and token >= 0 for token in prompt):
        return list(prompt)
    raise ValueError('prompt must be text or a nonempty list of token IDs')


def encode(model, messages, raw_prompt=None, enable_thinking=None, tools=None, tool_json=False, reasoning_effort=None):
    if getattr(model, "task", "generation") != "generation":
        raise ValueError("this model does not support generation")
    if raw_prompt is not None:
        if messages is not None or enable_thinking is not None or tools is not None or tool_json or reasoning_effort is not None:
            raise ValueError('raw prompts cannot include chat messages or template options')
        return model.raw_tokens(raw_prompt)
    options={'enable_thinking':enable_thinking} if enable_thinking is not None else {}
    if reasoning_effort is not None:options['reasoning_effort']=reasoning_effort
    if tools is not None:options['tools']=tools
    if tool_json:options['tool_json']=True
    return model.chat_tokens(messages,**options)


def decode_output(model, tokens):
    """Generated continuations may preserve space that full-text decoding strips."""
    tokenizer = model.tokenizer
    decode = getattr(tokenizer, 'decode_generated', tokenizer.decode)
    return decode(tokens, skip_special_tokens=True)
