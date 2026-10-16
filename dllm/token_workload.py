"""Stable identity for ordered integer-token qualification workloads."""
import hashlib
import json


def token_workload_digest(prompts, output_tokens):
    if (not isinstance(prompts,list) or not prompts
            or type(output_tokens) is not int or output_tokens<=0
            or any(not isinstance(ids,list) or not ids
                   or any(type(token) is not int or token<0 for token in ids)
                   for ids in prompts)):
        raise ValueError('workload identity requires nonempty token prompts and a positive output limit')
    payload=dict(version=1,prompts=prompts,output_tokens=output_tokens)
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()
