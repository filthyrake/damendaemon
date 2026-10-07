"""Owned equal-length CPU reference batches, independent of model family."""
import torch


def prepare_batch(text, conversations, max_new_tokens, max_context, **options):
    """Validate the entire batch before any model payload/state work."""
    if not isinstance(conversations, list) or not 1 <= len(conversations) <= 16:
        raise ValueError('reference batch requires 1..16 conversations')
    limits = ([max_new_tokens] * len(conversations)
              if type(max_new_tokens) is int else max_new_tokens)
    if (not isinstance(limits, list) or len(limits) != len(conversations)
            or any(type(n) is not int or not 1 <= n <= 4096 for n in limits)):
        raise ValueError('reference batch output limits require 1..4096 tokens per row')
    prompts = [text.chat_tokens(messages, **options) for messages in conversations]
    if not prompts[0] or any(len(row) != len(prompts[0]) for row in prompts):
        raise ValueError('reference batch requires equal encoded prompt lengths')
    if len(prompts[0]) + max(limits) > max_context:
        raise ValueError('reference batch prompt and requested output exceed context')
    return prompts, list(limits)


def generate_prepared_batch(runner, prompts, limits, streams, *, mapped_vocab_size, prefill, decode):
    """Internal lifecycle; family adapters validate prompts and construct parsers."""
    prompt_length = len(prompts[0])
    active = list(range(len(prompts)))
    finals = []
    state = None
    primary = None
    try:
        with runner.lock:
            runner.cleanup()
            logits, state = prefill(torch.tensor(prompts, dtype=torch.int64, device='cpu'),
                capacity=prompt_length + max(limits), max_cache_bytes=runner.max_cache_bytes)
        for step in range(max(limits)):
            if (not isinstance(logits, torch.Tensor) or logits.shape != (len(active), runner.text.vocab_size)
                    or logits.device.type != 'cpu' or logits.dtype != torch.float32):
                raise ValueError('invalid reference batch generation logits')
            mapped = logits[:, :mapped_vocab_size]
            if not torch.isfinite(mapped).all():
                raise ValueError('nonfinite reference batch generation logits')
            tokens = mapped.argmax(-1).tolist()
            surviving_slots = []
            surviving_rows = []
            next_tokens = []
            for slot, (row, token) in enumerate(zip(active, tokens)):
                for event in streams[row].feed(token):
                    yield dict(event, row=row)
                stopped = token in runner.text.eos
                finished = stopped or step + 1 == limits[row]
                if finished:
                    for event in streams[row].finish(truncated=not stopped):
                        yield dict(event, row=row)
                    finals.append(dict(row=row, finish_reason='stop' if stopped else 'length',
                        usage=dict(prompt_tokens=prompt_length, completion_tokens=step + 1)))
                else:
                    surviving_slots.append(slot)
                    surviving_rows.append(row)
                    next_tokens.append([token])
            if not surviving_rows:
                break
            if len(surviving_rows) != len(active):
                state.reorder(torch.tensor(surviving_slots, dtype=torch.int64, device='cpu'))
            active = surviving_rows
            logits = decode(torch.tensor(next_tokens, dtype=torch.int64, device='cpu'),
                                        state, position=prompt_length + step)
    except BaseException as error:
        primary = error
        raise
    finally:
        if state is not None:
            try:
                runner._release(state)
            except BaseException:
                if primary is None:
                    raise
                primary.add_note('reference batch cache cleanup failed; retained for retry before another prefill')
    # Report success only after every owned row's cache has been released.
    yield from sorted(finals, key=lambda event: event['row'])
