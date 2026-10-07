"""Independent V4 text encoding from the pinned MIT-licensed format specification.

Reference: deepseek-ai/DeepSeek-V4-Flash, revision
60d8d70770c6776ff598c94bb586a859a38244f1, encoding/encoding_dsv4.py.
Tool histories use deepseek_tool_prompt; internal tasks and developer/reminder roles remain unsupported.
"""
import json
from pathlib import Path

from tokenizers import Tokenizer

from .checkpoint import _unique_fields

BOS='<｜begin▁of▁sentence｜>'
EOS='<｜end▁of▁sentence｜>'
USER='<｜User｜>'
ASSISTANT='<｜Assistant｜>'
THINK='<think>'
END_THINK='</think>'
MAX_TEXT_BYTES=1024*1024


def render_chat(messages, *, enable_thinking=False, drop_thinking=True, reasoning_effort=None, tools=None):
    """Render ordinary complete text turns followed by a generation prefix."""
    if tools is not None:
        from .deepseek_tool_prompt import render_tool_chat
        return render_tool_chat(messages,tools,enable_thinking=enable_thinking,
                                drop_thinking=drop_thinking,reasoning_effort=reasoning_effort)
    if (type(enable_thinking) is not bool or type(drop_thinking) is not bool
            or reasoning_effort not in (None,'high')):
        raise ValueError('unsupported V4 text thinking options')
    if not isinstance(messages,list) or not 1 <= len(messages) <= 4096:
        raise ValueError('V4 text chat requires 1..4096 messages')
    parts=[BOS]
    expected='user'
    consumed=0
    for index,message in enumerate(messages):
        if not isinstance(message,dict) or set(message)-{'role','content','reasoning_content'}:
            raise ValueError('unsupported V4 text message fields')
        role,content=message.get('role'),message.get('content')
        reasoning=message.get('reasoning_content','')
        if not isinstance(content,str) or not isinstance(reasoning,str) or (role != 'assistant' and 'reasoning_content' in message):
            raise ValueError('V4 messages require text content and assistant-only reasoning')
        consumed+=len(content.encode('utf-8'))+len(reasoning.encode('utf-8'))
        if consumed > MAX_TEXT_BYTES:
            raise ValueError('V4 conversation exceeds text byte limit')
        if role == 'system' and index == 0:
            parts.append(content)
            continue
        if role != expected:
            raise ValueError('V4 text chat requires alternating user/assistant turns and a final user')
        if role == 'user':
            parts.extend((USER,content,ASSISTANT))
            keep=enable_thinking and (not drop_thinking or index == len(messages)-1)
            parts.append(THINK if keep else END_THINK)
            expected='assistant'
        else:
            if enable_thinking and not drop_thinking:
                parts.extend((reasoning,END_THINK))
            parts.extend((content,EOS))
            expected='user'
    if expected != 'assistant':
        raise ValueError('V4 generation requires a final user message')
    return ''.join(parts)


def _read(path,limit):
    with Path(path).open('rb') as reader:
        value=reader.read(limit+1)
    if len(value) > limit:
        raise ValueError('V4 tokenizer metadata exceeds byte limit')
    return value.decode('utf-8')


class DeepSeekText:
    """Local tokenizer adapter, separate from model/HTTP admission."""
    def __init__(self,path,*,vocab_size):
        if type(vocab_size) is not int or not 1 <= vocab_size <= 2**21:
            raise ValueError('invalid V4 tokenizer vocabulary size')
        path=Path(path)
        config=json.loads(_read(path/'tokenizer_config.json',1024*1024),object_pairs_hook=_unique_fields)
        if (not isinstance(config,dict) or config.get('add_bos_token') is not False
                or config.get('add_eos_token') is not False or config.get('clean_up_tokenization_spaces') is not False
                or config.get('tokenizer_class') != 'PreTrainedTokenizerFast'):
            raise ValueError('unsupported V4 tokenizer configuration')
        for name,expected in (('bos_token',BOS),('eos_token',EOS),('pad_token',EOS)):
            value=config.get(name)
            if isinstance(value,dict):value=value.get('content')
            if value != expected:
                raise ValueError('V4 tokenizer special-token configuration mismatch')
        tokenizer=Tokenizer.from_str(_read(path/'tokenizer.json',64*1024*1024))
        tokenizer.no_padding();tokenizer.no_truncation()
        vocabulary=tokenizer.get_vocab()
        if len(vocabulary) != vocab_size or set(vocabulary.values()) != set(range(vocab_size)):
            raise ValueError('V4 tokenizer vocabulary differs from model')
        for special in (BOS,EOS,USER,ASSISTANT,THINK,END_THINK):
            token=tokenizer.token_to_id(special)
            if token is None or tokenizer.encode(special,add_special_tokens=False).ids != [token]:
                raise ValueError('V4 control marker must be exactly one tokenizer token')
        self.tokenizer=tokenizer
        self.vocab_size=vocab_size
        self.eos={tokenizer.token_to_id(EOS)}

    def raw_tokens(self,prompt):
        if isinstance(prompt,str):
            if len(prompt.encode('utf-8')) > MAX_TEXT_BYTES:
                raise ValueError('V4 raw prompt exceeds byte limit')
            return self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if isinstance(prompt,list) and prompt and all(type(t) is int and 0 <= t < self.vocab_size for t in prompt):
            return list(prompt)
        raise ValueError('V4 raw prompt must be text or valid nonempty token IDs')

    def chat_tokens(self,messages,**options):
        rendered=render_chat(messages,**options)
        return self.tokenizer.encode(rendered,add_special_tokens=False).ids

    def decode(self,tokens,*,skip_special_tokens=True):
        if not isinstance(tokens,list) or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise ValueError('invalid V4 output token IDs')
        return self.tokenizer.decode(tokens,skip_special_tokens=skip_special_tokens)
