"""Bounded raw tokenizer input for the explicit CPU completion command."""
import json
from pathlib import Path
from tokenizers import Tokenizer
from .safetensor_header import _unique_fields
from .cpu_pipeline_generation import validate_request


def read_json(path,limit):
    with Path(path).open('rb') as reader:raw=reader.read(limit+1)
    if len(raw)>limit:raise ValueError('CPU completion input exceeds metadata byte limit')
    return json.loads(raw,object_pairs_hook=_unique_fields)


class CpuReferenceText:
    def __init__(self,path,config):
        if type(config.get('vocab_size')) is not int or not 1<=config['vocab_size']<=2**21:
            raise ValueError('invalid CPU model vocabulary')
        path=Path(path)
        with (path/'tokenizer.json').open('rb') as reader:raw=reader.read(32*1024**2+1)
        if len(raw)>32*1024**2:raise ValueError('CPU tokenizer exceeds 32 MiB limit')
        try:self.tokenizer=Tokenizer.from_str(raw.decode('utf-8'))
        except Exception as error:raise ValueError('invalid CPU reference tokenizer') from error
        vocab=self.tokenizer.get_vocab();ids=set(vocab.values())
        if not vocab or len(vocab)>2**21 or ids!=set(range(len(vocab))) or len(vocab)>config.get('vocab_size',0):
            raise ValueError('CPU tokenizer requires a contiguous mapped vocabulary within the model')
        self.vocab_size=len(vocab);self.tokenizer.no_truncation();self.tokenizer.no_padding()
        generation=read_json(path/'generation_config.json',1024**2) if (path/'generation_config.json').exists() else {}
        if not isinstance(generation,dict):raise ValueError('invalid CPU generation configuration')
        eos=generation.get('eos_token_id',config.get('eos_token_id',[]))
        if type(eos) is int:eos=[eos]
        if not isinstance(eos,list) or len(eos)>64 or any(type(t) is not int or not 0<=t<self.vocab_size for t in eos):
            raise ValueError('invalid CPU EOS configuration')
        self.eos_ids=frozenset(eos)
        self.excluded_ids=frozenset(vocab[t] for t in ('<|image_pad|>','<|video_pad|>','<|vision_start|>','<|vision_end|>') if t in vocab)
        for key in ('image_token_id','video_token_id','vision_start_token_id','vision_end_token_id'):
            if key in config:
                value=config[key]
                if type(value) is not int or not 0<=value<self.vocab_size:raise ValueError('invalid CPU media token configuration')
                self.excluded_ids=self.excluded_ids|{value}
        if self.excluded_ids&self.eos_ids:raise ValueError('CPU EOS cannot be a media token')

    def tokens(self,prompt,*,max_context,max_new_tokens,chunk_size):
        if isinstance(prompt,str):
            if not prompt or len(prompt.encode('utf-8'))>1024**2:raise ValueError('CPU raw prompt must contain at most 1 MiB text')
            tokens=self.tokenizer.encode(prompt,add_special_tokens=True).ids
        else:tokens=prompt
        return validate_request(tokens,vocab_size=self.vocab_size,max_context=max_context,max_new_tokens=max_new_tokens,
                                chunk_size=chunk_size,eos_ids=self.eos_ids,excluded_ids=self.excluded_ids)

    def decode(self,tokens):return self.tokenizer.decode(tokens,skip_special_tokens=True)
