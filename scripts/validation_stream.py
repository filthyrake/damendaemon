"""Bounded client-side evidence for disconnecting a nonterminal HTTP stream."""
import json
from dllm.bench import events


def consume_nonterminal_prefix(response, count=2):
    """Read count nonterminal events; caller must close its response afterwards.

    This proves only that the client disconnected before observing completion.
    Server-side cancellation and overlap need separate measurements.
    """
    if type(count) is not int or count < 1:
        raise ValueError('cancellation prefix count must be a positive integer')
    stream = events(response, require_terminated=True)
    for _ in range(count):
        try:
            event = next(stream)
        except StopIteration as error:
            raise ValueError('stream ended before cancellation prefix') from error
        if event == '[DONE]':
            raise ValueError('stream completed before cancellation')
        payload = json.loads(event)
        choices = payload.get('choices') if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError('cancellation prefix requires nonterminal choices')
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('finish_reason') is not None:
                raise ValueError('stream finished before cancellation')
