"""Owned equal-length CPU reference batches, independent of model family."""
from contextlib import closing
import torch
from .reference_cancellation import check_cancelled


def prepare_batch(text, conversations, max_new_tokens, max_context, *, tokenize=None, ragged=False, **options):
    """Validate the entire batch before any model payload/state work."""
    if type(ragged) is not bool:raise ValueError('ragged batch option must be boolean')
    if not isinstance(conversations, list) or not 1 <= len(conversations) <= 16:
        raise ValueError('reference batch requires 1..16 conversations')
    limits = ([max_new_tokens] * len(conversations)
              if type(max_new_tokens) is int else max_new_tokens)
    if (not isinstance(limits, list) or len(limits) != len(conversations)
            or any(type(n) is not int or not 1 <= n <= 4096 for n in limits)):
        raise ValueError('reference batch output limits require 1..4096 tokens per row')
    encode = text.chat_tokens if tokenize is None else tokenize
    prompts = [encode(messages, **options) for messages in conversations]
    if any(not row for row in prompts):raise ValueError('reference batch needs nonempty encoded prompts')
    if not ragged and any(len(row) != len(prompts[0]) for row in prompts):
        raise ValueError('reference batch requires equal encoded prompt lengths')
    if any(len(row)+limit>max_context for row,limit in zip(prompts,limits)):
        raise ValueError('reference batch prompt and requested output exceed context')
    return prompts, list(limits)


def _generate_equal_batch(runner, prompts, limits, streams, *, mapped_vocab_size, prefill, decode, finalize=None):
    """Internal lifecycle; family adapters validate prompts and construct parsers."""
    prompt_length = len(prompts[0])
    active = list(range(len(prompts)))
    finals = []
    state = None
    primary = None
    try:
        check_cancelled()
        with runner.lock:
            runner.cleanup()
            logits, state = prefill(torch.tensor(prompts, dtype=torch.int64, device='cpu'),
                capacity=prompt_length + max(limits), max_cache_bytes=runner.max_cache_bytes)
        for step in range(max(limits)):
            check_cancelled()
            if (not isinstance(logits, torch.Tensor) or logits.shape != (len(active), runner.text.vocab_size)
                    or logits.device.type != 'cpu' or logits.dtype != torch.float32):
                raise RuntimeError('invalid reference batch generation logits')
            mapped = logits[:, :mapped_vocab_size]
            if not torch.isfinite(mapped).all():
                raise RuntimeError('nonfinite reference batch generation logits')
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
                    check_cancelled()
                    events = streams[row].finish(truncated=not stopped)
                    reason = 'stop' if stopped else 'length'
                    if finalize is not None:
                        reason = finalize(row, events, reason)
                    for event in events:
                        yield dict(event, row=row)
                    finals.append(dict(row=row, finish_reason=reason,
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
            check_cancelled()
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
    check_cancelled()
    yield from sorted(finals, key=lambda event: event['row'])


def generate_prepared_batch(runner, prompts, limits, streams, *, mapped_vocab_size, prefill, decode, finalize=None):
    """Run stable equal-length cohorts sequentially, with one live cache budget.

    Entire-batch validation belongs to prepare_batch. No success is published
    until all cohorts release their caches, even if later execution fails.
    """
    cohorts={}
    for row,prompt in enumerate(prompts):cohorts.setdefault(len(prompt),[]).append(row)
    finals=[]
    for rows in cohorts.values():
        finish=None if finalize is None else lambda row,events,reason:finalize(rows[row],events,reason)
        generation=_generate_equal_batch(runner,[prompts[row] for row in rows],[limits[row] for row in rows],
            [streams[row] for row in rows],mapped_vocab_size=mapped_vocab_size,prefill=prefill,decode=decode,finalize=finish)
        with closing(generation):
            for event in generation:
                event=dict(event,row=rows[event['row']])
                if 'finish_reason' in event:finals.append(event)
                else:yield event
    check_cancelled()
    yield from sorted(finals,key=lambda event:event['row'])
