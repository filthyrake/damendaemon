"""Inspect equivalent tokenizer decoder sequences without recursive traversal."""
import json


def decoder_steps(decoder):
    if decoder is None:return []
    steps=[];pending=[json.loads(decoder.__getstate__())]
    while pending:
        step=pending.pop()
        if step.get('type')=='Sequence':
            pending.extend(reversed(step['decoders']))
        else:steps.append(step)
    return steps


def is_bytelevel(decoder):
    """True only for one ByteLevel step, allowing transparent Sequence wrappers."""
    steps=decoder_steps(decoder)
    return len(steps)==1 and steps[0].get('type')=='ByteLevel'
