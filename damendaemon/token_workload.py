"""Stable identity for ordered integer-token qualification workloads."""
import hashlib
import json
from pathlib import Path


def load_token_prompt(path, *, max_bytes=16*1024*1024, max_tokens=2**20):
    """Read a bounded UTF-8 token array before starting comparison work."""
    if type(max_bytes) is not int or max_bytes<1 or type(max_tokens) is not int or max_tokens<1:
        raise ValueError('invalid token prompt limits')
    with Path(path).open('rb') as source:
        raw=source.read(max_bytes+1)
    if len(raw)>max_bytes:raise ValueError('token prompt exceeds byte limit')
    try:tokens=json.loads(raw.decode('utf-8'))
    except RecursionError:raise ValueError('token prompt nesting is too deep') from None
    if (not isinstance(tokens,list) or not 1<=len(tokens)<=max_tokens
            or any(type(token) is not int or token<0 for token in tokens)):
        raise ValueError('token prompt requires a bounded nonempty nonnegative integer array')
    return tokens


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


def teacher_token_lookup(teacher, prompts, output_tokens, vocab_size):
    """Validate complete eager/graph teacher inputs before allocating a model."""
    identity=token_workload_digest(prompts,output_tokens)
    if type(vocab_size) is not int or vocab_size<=0:raise ValueError('invalid teacher vocabulary size')
    if teacher is None:return {}
    if not isinstance(teacher,dict) or teacher.get('workload_sha256')!=identity:
        raise ValueError('teacher workload identity is missing or differs')
    cases=teacher.get('cases')
    if not isinstance(cases,list) or len(cases)!=2*len(prompts):
        raise ValueError('teacher trace must cover every eager and graph prompt')
    result={}
    for case in cases:
        if not isinstance(case,dict):raise ValueError('invalid teacher case')
        graph,index=case.get('graph'),case.get('case')
        if type(graph) is not bool or type(index) is not int or not 0<=index<len(prompts):
            raise ValueError('invalid teacher case identity')
        key=(graph,index)
        if key in result:raise ValueError('duplicate teacher case identity')
        tokens=case.get('reference_tokens')
        if (type(case.get('prompt_tokens')) is not int or case['prompt_tokens']!=len(prompts[index])
                or not isinstance(tokens,list) or len(tokens)!=output_tokens
                or any(type(token) is not int or not 0<=token<vocab_size for token in tokens)):
            raise ValueError('invalid teacher prompt length or output tokens')
        result[key]=list(tokens)
    return result
