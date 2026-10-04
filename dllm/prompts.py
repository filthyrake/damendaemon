"""Shared raw-prompt validation and chat/raw dispatch."""


def validate_raw(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and all(type(token) is int and token >= 0 for token in prompt):
        return list(prompt)
    raise ValueError('prompt must be text or a nonempty list of token IDs')


def encode(model, messages, raw_prompt=None, enable_thinking=None):
    if getattr(model, "task", "generation") != "generation":
        raise ValueError("this model supports embeddings, not generation")
    if raw_prompt is not None:
        if messages is not None or enable_thinking is not None:
            raise ValueError('raw prompts cannot include chat messages or template options')
        return model.raw_tokens(raw_prompt)
    return model.chat_tokens(messages, **(
        {'enable_thinking': enable_thinking} if enable_thinking is not None else {}))
