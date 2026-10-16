"""Independent ordinary GLM-5.3 text format and local tokenizer adapter.

Reference: zai-org/GLM-5.3 at aca966e4e02791568aa6a4ced368624b3d897f42,
chat_template.jinja and tokenizer/generation metadata. See docs/licenses/glm53-LICENSE.txt.
The pinned template always opens a reasoning channel; no non-thinking mode is invented.
"""
from pathlib import Path
from tokenizers import Tokenizer
from .safetensor_header import parse_metadata_json

MASK='[gMASK]'
SOP='<sop>'
SYSTEM='<|system|>'
USER='<|user|>'
ASSISTANT='<|assistant|>'
EOS='<|endoftext|>'
OBSERVATION='<|observation|>'
THINK='<think>'
END_THINK='</think>'
CONTROLS=(MASK,SOP,SYSTEM,USER,ASSISTANT,EOS,OBSERVATION,THINK,END_THINK)
MAX_TEXT_BYTES=1024**2
MAX_TOKENS=2**20


def render_chat(messages,*,reasoning_effort=None,clear_thinking=False):
    """Render complete ordinary text turns and the pinned reasoning prefix."""
    if reasoning_effort not in (None,'low','high','max') or type(clear_thinking) is not bool:
        raise ValueError('unsupported GLM reasoning options')
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:
        raise ValueError('GLM chat requires 1..4096 ordinary text messages')
    parts=[MASK,SOP,SYSTEM,'Reasoning Effort: ',(reasoning_effort or 'max').capitalize()]
    expected='user';consumed=0
    for index,message in enumerate(messages):
        if not isinstance(message,dict) or set(message)-{'role','content','reasoning_content'}:
            raise ValueError('unsupported GLM text message fields')
        role,content=message.get('role'),message.get('content')
        reasoning=message.get('reasoning_content')
        if (not isinstance(content,str) or ('reasoning_content' in message and
                (role!='assistant' or not isinstance(reasoning,str)))):
            raise ValueError('GLM requires text content and assistant-only text reasoning')
        consumed+=len(content.encode('utf-8'))+len((reasoning or '').encode('utf-8'))
        if consumed>MAX_TEXT_BYTES:raise ValueError('GLM conversation exceeds text byte limit')
        if role=='system' and index==0:
            parts.extend((SYSTEM,content));continue
        if role!=expected:raise ValueError('GLM text chat requires alternating turns and a final user')
        if role=='user':parts.extend((USER,content));expected='assistant'
        else:
            if reasoning is None and END_THINK in content:
                reasoning=content.split(END_THINK)[0].split(THINK)[-1]
                content=content.split(END_THINK)[-1]
            parts.extend((ASSISTANT,THINK,'' if clear_thinking else reasoning or '',END_THINK,content.strip()))
            expected='user'
    if expected!='assistant':raise ValueError('GLM generation requires a final user message')
    parts.extend((ASSISTANT,THINK));rendered=''.join(parts)
    if len(rendered.encode('utf-8'))>MAX_TEXT_BYTES:raise ValueError('GLM rendered prompt exceeds text byte limit')
    return rendered


def _read(path,limit):
    with Path(path).open('rb') as reader:raw=reader.read(limit+1)
    if len(raw)>limit:raise ValueError('GLM tokenizer metadata exceeds byte limit')
    return raw.decode('utf-8')


class GlmText:
    """Tokenize locally without remote code; keep padded model rows distinct."""
    def __init__(self,path,*,vocab_size):
        if type(vocab_size) is not int or not 1<=vocab_size<=2**21:
            raise ValueError('invalid GLM model vocabulary size')
        path=Path(path)
        config=parse_metadata_json(_read(path/'tokenizer_config.json',1024**2))
        required={'backend':'tokenizers','tokenizer_class':'TokenizersBackend','clean_up_tokenization_spaces':False,
                  'do_lower_case':False,'remove_space':False,'eos_token':EOS,'pad_token':EOS}
        if not isinstance(config,dict) or any(config.get(k)!=v or type(config.get(k)) is not type(v) for k,v in required.items()):
            raise ValueError('unsupported GLM tokenizer configuration')
        tokenizer=Tokenizer.from_str(_read(path/'tokenizer.json',64*1024**2))
        tokenizer.no_padding();tokenizer.no_truncation()
        vocabulary=tokenizer.get_vocab();count=len(vocabulary)
        if not 1<=count<=vocab_size or set(vocabulary.values())!=set(range(count)):
            raise ValueError('GLM tokenizer vocabulary must be contiguous within model rows')
        for marker in CONTROLS:
            token=tokenizer.token_to_id(marker)
            if token is None or tokenizer.encode(marker,add_special_tokens=False).ids!=[token]:
                raise ValueError('GLM control marker must be exactly one tokenizer token')
        generation=parse_metadata_json(_read(path/'generation_config.json',1024**2))
        stops={tokenizer.token_to_id(marker) for marker in (EOS,USER,OBSERVATION)}
        eos=generation.get('eos_token_id') if isinstance(generation,dict) else None
        if (not isinstance(eos,list) or any(type(v) is not int for v in eos) or len(eos)!=len(stops) or set(eos)!=stops
                or type(generation.get('pad_token_id')) is not int or generation['pad_token_id']!=tokenizer.token_to_id(EOS)):
            raise ValueError('GLM generation stop/pad markers differ from tokenizer')
        self.tokenizer=tokenizer;self.vocab_size=vocab_size;self.tokenizer_size=count;self.eos=stops

    def raw_tokens(self,prompt):
        if isinstance(prompt,str):
            if len(prompt.encode('utf-8'))>MAX_TEXT_BYTES:raise ValueError('GLM raw prompt exceeds byte limit')
            return self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if (isinstance(prompt,list) and 1<=len(prompt)<=MAX_TOKENS
                and all(type(v) is int and 0<=v<self.tokenizer_size for v in prompt)):
            return list(prompt)
        raise ValueError('GLM raw prompt requires text or mapped nonempty token IDs')

    def chat_tokens(self,messages,**options):
        return self.tokenizer.encode(render_chat(messages,**options),add_special_tokens=False).ids

    def decode(self,tokens,*,skip_special_tokens=True):
        if (not isinstance(tokens,list) or len(tokens)>MAX_TOKENS
                or any(type(v) is not int or not 0<=v<self.tokenizer_size for v in tokens)):
            raise ValueError('invalid or unmapped GLM output token IDs')
        return self.tokenizer.decode(tokens,skip_special_tokens=skip_special_tokens)
