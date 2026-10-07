"""Bounded raw completion input files shared by explicit CPU commands."""
from .cpu_reference_text import read_json


def completion_inputs(text,*,prompt=None,tokens_file=None,batch_file=None,max_context=512,
                      max_new_tokens=128,chunk_size=128):
    if sum(value is not None for value in (prompt,tokens_file,batch_file))!=1:
        raise ValueError('choose exactly one raw prompt, token file or batch file')
    if batch_file is not None:
        requests=read_json(batch_file,1024**2)
        if not isinstance(requests,list) or not 1<=len(requests)<=4:raise ValueError('CPU batch needs 1..4 requests')
        tokens=[];limits=[]
        for request in requests:
            if (not isinstance(request,dict) or set(request)-{'prompt','tokens','max_new_tokens'}
                    or ('prompt' in request)==('tokens' in request)
                    or ('prompt' in request and not isinstance(request['prompt'],str))
                    or ('tokens' in request and not isinstance(request['tokens'],list))):
                raise ValueError('CPU batch requests need exactly one raw prompt or token list')
            limit=request.get('max_new_tokens',max_new_tokens);limits.append(limit)
            tokens.append(text.tokens(request.get('prompt',request.get('tokens')),max_context=max_context,
                                      max_new_tokens=limit,chunk_size=chunk_size))
    else:
        prompt=prompt if prompt is not None else read_json(tokens_file,1024**2)
        if tokens_file is not None and not isinstance(prompt,list):raise ValueError('token file must contain a JSON list')
        tokens=text.tokens(prompt,max_context=max_context,max_new_tokens=max_new_tokens,chunk_size=chunk_size)
        limits=max_new_tokens
    return tokens,limits
