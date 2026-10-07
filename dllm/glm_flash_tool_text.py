"""Pinned bounded Flash text/tool histories; no media or tool execution.

Tool-system prose and formatting derive from GLM-5.3-Flash chat_template.jinja,
revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a (MIT).
See docs/licenses/glm53-flash-MIT.txt.
"""
from .glm_flash_text import FlashText
from .glm_flash_tools import FlashTools, _dump, _json, _literal, _text
from .glm_text import MASK, SOP, SYSTEM, USER, ASSISTANT, OBSERVATION, THINK, END_THINK, MAX_TEXT_BYTES, MAX_TOKENS

TOOL_PREFIX='''
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
'''
TOOL_SUFFIX='''</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>'''


def _identity(value):
    _text(value,128)
    if not value or any(ord(c)<32 for c in value):raise ValueError('Flash tool result identity must be nonempty text')
    return value


def render_tool_chat(messages,tools,*,reasoning_effort=None,clear_thinking=False):
    if reasoning_effort not in (None,'low','high','max') or type(clear_thinking) is not bool:
        raise ValueError('unsupported Flash tool reasoning options')
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:
        raise ValueError('Flash tool history requires 1..4096 messages')
    codec=FlashTools(tools)
    declarations=_json(_dump(tools))
    schemas=[]
    for tool in declarations:
        encoded=_dump({key:value for key,value in tool['function'].items() if key!='strict'})
        _literal(encoded)
        if '<tools' in encoded or '</tools' in encoded:raise ValueError('reserved Flash tool declaration delimiter')
        schemas.append(encoded+'\n')
    parts=[];size=0
    def append(value):
        nonlocal size
        size+=len(value.encode('utf-8'))
        if size>MAX_TEXT_BYTES:raise ValueError('Flash tool prompt exceeds byte limit')
        parts.append(value)
    append(MASK+SOP+SYSTEM+'Reasoning Effort: '+(reasoning_effort or 'max').capitalize())
    append(SYSTEM+TOOL_PREFIX+''.join(schemas)+TOOL_SUFFIX)
    last_user=max((index for index,message in enumerate(messages)
                   if isinstance(message,dict) and message.get('role')=='user'),default=-1)
    expected='user';index=0
    while index<len(messages):
        message=messages[index]
        if not isinstance(message,dict):raise ValueError('Flash tool messages must be mappings')
        role=message.get('role')
        if role=='system' and index==0:
            if set(message)!={'role','content'}:raise ValueError('unsupported Flash system fields')
            append(SYSTEM+_literal(message['content']));index+=1;continue
        if role=='user' and expected in ('user','after_tools'):
            if set(message)!={'role','content'}:raise ValueError('unsupported Flash user fields')
            append(USER+_literal(message['content']));expected='assistant';index+=1;continue
        if role!='assistant' or expected not in ('assistant','after_tools'):
            raise ValueError('invalid Flash tool conversation order')
        if set(message)-{'role','content','reasoning_content','tool_calls'}:
            raise ValueError('unsupported Flash assistant fields')
        content=message.get('content')
        content='' if content is None else _literal(content)
        reasoning=_literal(message['reasoning_content']) if 'reasoning_content' in message else ''
        append(ASSISTANT+THINK+(reasoning if not clear_thinking or index>last_user else '')+END_THINK+content.strip())
        calls=message.get('tool_calls')
        if calls is None:
            expected='user';index+=1;continue
        rendered=codec.render(calls)
        identities=[_identity(call.get('id')) for call in calls]
        if len(set(identities))!=len(identities):raise ValueError('duplicate Flash tool call identity')
        append(rendered);index+=1;results={}
        while index<len(messages) and isinstance(messages[index],dict) and messages[index].get('role')=='tool':
            result=messages[index]
            if set(result)!={'role','content','tool_call_id'}:raise ValueError('unsupported Flash tool result fields')
            identity=_identity(result['tool_call_id'])
            if identity not in identities or identity in results:raise ValueError('foreign or duplicate Flash tool result identity')
            results[identity]=_literal(result['content']);index+=1
        if set(results)!=set(identities):raise ValueError('missing Flash tool results')
        append(OBSERVATION)
        for identity in identities:append('<tool_response>'+results[identity]+'</tool_response>')
        expected='after_tools'
    if expected not in ('assistant','after_tools'):raise ValueError('Flash generation needs a final user or complete tool results')
    append(ASSISTANT+THINK)
    return ''.join(parts)


class FlashToolText(FlashText):
    """Explicit tool-history entry point; ordinary chat_tokens remains unchanged."""
    def tool_chat_tokens(self,messages,tools,**options):
        prompt=render_tool_chat(messages,tools,**options)
        tokens=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if not tokens or len(tokens)>MAX_TOKENS:raise ValueError('Flash tool prompt token count exceeds bounds')
        return tokens
