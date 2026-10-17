"""Direct next-token choices, conditional on an explicit candidate set.

Inspired by typed-decision interfaces (TypeSafe Jev) and indexed output heads
(NInfer); independently implemented. These are not calibrated confidence scores
and do not implement a trained Jev decision head or multi-token answer scoring.
"""
from .prompts import validate_raw


def options(prompt,candidates,*,max_length=None):
    prompt=validate_raw(prompt,max_length=max_length)
    if (not isinstance(candidates,list) or not 2<=len(candidates)<=256
            or any(type(token) is not int or token<0 for token in candidates)
            or len(set(candidates))!=len(candidates)):
        raise ValueError('candidate_token_ids must contain 2 to 256 distinct nonnegative integers')
    return prompt,list(candidates)
