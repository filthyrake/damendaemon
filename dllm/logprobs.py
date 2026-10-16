"""Opt-in raw model log probabilities and tokenizer-aware token presentation."""
from collections import OrderedDict
import math
import os
import re


def options(enabled=False,top=0):
    enabled=False if enabled is None else enabled
    top=0 if top is None else top
    if type(enabled) is not bool or type(top) is not int or not 0<=top<=20 or (top and not enabled):
        raise ValueError('logprobs must be boolean; top_logprobs requires logprobs and must be 0..20')
    return top if enabled else None


def scores(logits,selected,top):
    """Score original logits, never sampling-adjusted or graph-mutated storage."""
    import torch
    selected=torch.as_tensor(selected,device=logits.device,dtype=torch.long).reshape(1)
    return score_rows(logits.unsqueeze(0),selected,[top])[0]


def score_rows(logits,selected,tops):
    """One set of reductions/readbacks for a scored batch, not per request."""
    import torch
    backend=os.environ.get('DLLM_LOGPROBS_KERNELS','auto')
    if backend not in ('torch','triton','auto'):
        raise ValueError('DLLM_LOGPROBS_KERNELS must be auto, torch or triton')
    native_case=backend=='triton' or len(tops)>1 or max(tops)>0
    if backend!='torch' and native_case and logits.device.type=='cuda' and logits.shape[-1]<=262144:
        from .logprob_kernels import score_rows as native
        return native(logits,selected,tops)
    if backend=='triton':
        raise ValueError('native logprobs require CUDA and vocabulary <= 262144')
    values=torch.log_softmax(logits.float(),dim=-1)
    best,ids=values.topk(min(max(tops),values.shape[-1]),dim=-1)
    chosen=selected.reshape(-1,1)
    valid=(chosen>=0)&(chosen<values.shape[-1])
    chosen_values=values.gather(1,chosen.clamp(0,values.shape[-1]-1)).masked_fill(~valid,float('nan'))
    numbers=torch.cat((chosen_values,best),dim=1).cpu().tolist()
    token_ids=torch.cat((chosen,ids),dim=1).cpu().tolist()
    result=[]
    for row,ids,top in zip(numbers,token_ids,tops):
        if any(not math.isfinite(value) for value in row):
            raise RuntimeError('cannot report non-finite model log probabilities')
        result.append({'token_id':ids[0],'logprob':row[0],'top':list(zip(ids[1:top+1],row[1:top+1]))})
    return result


class TokenText:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.cache=OrderedDict()
        from .tokenizer_decoders import decoder_steps
        steps=decoder_steps(tokenizer.decoder)
        self.byte_level=any(step.get('type')=='ByteLevel' for step in steps)
        self.byte_fallback=any(step.get('type')=='ByteFallback' for step in steps)
        self.metaspace=next((step.get('replacement','▁') for step in steps if step.get('type')=='Metaspace'),None)
        self.replacements=[(step['pattern']['String'],step['content']) for step in steps
                           if step.get('type')=='Replace' and 'String' in step.get('pattern',{})]
        self.added=tokenizer.get_added_tokens_decoder()
        # ByteLevel maps printable bytes directly and the rest above U+0100.
        printable=list(range(33,127))+list(range(161,173))+list(range(174,256))
        missing=[value for value in range(256) if value not in printable]
        self.byte_values={chr(value):value for value in printable}
        self.byte_values.update({chr(256+i):value for i,value in enumerate(missing)})

    def token(self,token_id):
        if token_id not in self.cache:
            value=self._token(token_id)
            if len(self.cache)>=1024:self.cache.popitem(last=False)
            self.cache[token_id]=value
        self.cache.move_to_end(token_id)
        return self.cache[token_id]

    def _token(self,token_id):
        if token_id in getattr(self.tokenizer,"suppressed_token_ids", ()):
            return "", ()
        piece=self.tokenizer.id_to_token(token_id)
        if piece is None:
            raise ValueError('token ID is absent from tokenizer')
        if token_id in self.added:
            added=self.added[token_id]
            if added.special:return piece,None
            # GGUF user pieces deliberately bypass the backend decoder.
            # Native added pieces follow their supported byte decoder instead.
            mapped=self.byte_level or self.byte_fallback or self.metaspace is not None
            if not mapped or token_id in getattr(self.tokenizer,'_literal_tokens',{}):
                return piece,tuple(piece.encode('utf-8'))
        raw=None
        if self.byte_level:
            raw=(bytes(self.byte_values[char] for char in piece)
                 if all(char in self.byte_values for char in piece) else piece.encode('utf-8'))
        elif self.byte_fallback or self.metaspace is not None:
            if self.byte_fallback and re.fullmatch(r'<0x[0-9A-Fa-f]{2}>',piece):
                raw=bytes([int(piece[3:5],16)])
            else:
                for old,new in self.replacements:piece=piece.replace(old,new)
                if self.metaspace is not None:piece=piece.replace(self.metaspace,' ')
                raw=piece.encode('utf-8')
        if raw is None:
            return self.tokenizer.decode([token_id],skip_special_tokens=False),None
        return raw.decode('utf-8',errors='replace'),tuple(raw)

    def format(self,value):
        def item(token_id,probability):
            token,raw=self.token(token_id)
            return {'token':token,'logprob':probability,'bytes':None if raw is None else list(raw)}
        return {**item(value['token_id'],value['logprob']),
                'top_logprobs':[item(token,probability) for token,probability in value['top']]}


def decoder_for(model):
    decoder=getattr(model,'logprob_decoder',None)
    if decoder is None:
        decoder=model.logprob_decoder=TokenText(model.tokenizer)
    return decoder
