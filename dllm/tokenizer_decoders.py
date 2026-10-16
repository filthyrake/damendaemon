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
