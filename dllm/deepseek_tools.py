"""Bounded independent DSML codec; never executes a tool.

Wire format: MIT-licensed deepseek-ai/DeepSeek-V4-Flash, revision
60d8d70770c6776ff598c94bb586a859a38244f1, encoding/encoding_dsv4.py.
This strict subset requires JSON object arguments and unambiguous names/values.
Call IDs are application metadata and are absent from the DSML wire format.
"""
import json
import math
import re

from .checkpoint import _unique_fields
from .deepseek_text import MAX_TEXT_BYTES, BOS, EOS, USER, ASSISTANT, THINK, END_THINK

DSML='｜DSML｜'
START=f'<{DSML}tool_calls>\n'
END=f'</{DSML}tool_calls>'
INVOKE_END=f'</{DSML}invoke>'
PARAM_END=f'</{DSML}parameter>'
INVOKE=re.compile(r'<｜DSML｜invoke name="([^"\n]*)">\n')
PARAM=re.compile(r'<｜DSML｜parameter name="([^"\n]*)" string="(true|false)">')
MAX_CALLS=64
MAX_PARAMETERS=256
CONTROLS=(DSML,BOS,EOS,USER,ASSISTANT,THINK,END_THINK)


def _text(value):
    if not isinstance(value,str):raise ValueError('DSML requires text')
    try:size=len(value.encode('utf-8'))
    except UnicodeError:raise ValueError('invalid DSML Unicode') from None
    if size > MAX_TEXT_BYTES:raise ValueError('DSML exceeds byte limit')
    return size


def _name(value):
    if (not isinstance(value,str) or not 1 <= len(value) <= 128
            or any(ord(c)<32 or c in '\"<>\x7f' for c in value)
            or any(marker in value for marker in CONTROLS)):
        raise ValueError('unsupported DSML name')
    _text(value)
    return value


def _float(value):
    result=float(value)
    if not math.isfinite(result):raise ValueError('nonfinite tool argument')
    return result


def _constant(value):
    raise ValueError('nonfinite tool argument')


def _json(value):
    _text(value)
    try:
        return json.loads(value,object_pairs_hook=_unique_fields,
                          parse_float=_float,parse_constant=_constant)
    except (ValueError,RecursionError):
        raise ValueError('tool arguments require finite, unambiguous JSON') from None


def _dump(value):
    try:return json.dumps(value,ensure_ascii=False,allow_nan=False)
    except (ValueError,TypeError,RecursionError):
        raise ValueError('unsupported tool argument value') from None


def _value(value):
    _text(value)
    if any(marker in value for marker in CONTROLS):
        raise ValueError('tool argument contains a reserved control marker')
    return value


def render_tool_calls(calls):
    """Encode 1..64 OpenAI-shaped calls, without the completion prefix or EOS."""
    if not isinstance(calls,list) or not 1 <= len(calls) <= MAX_CALLS:
        raise ValueError('DSML requires 1..64 tool calls')
    parts=[]
    size=0
    def append(value):
        nonlocal size
        size+=_text(value)
        if size > MAX_TEXT_BYTES:raise ValueError('DSML exceeds byte limit')
        parts.append(value)
    append(START)
    for index,call in enumerate(calls):
        if (not isinstance(call,dict) or set(call)-{'type','function','id'}
                or call.get('type') != 'function'):
            raise ValueError('unsupported tool call')
        function=call.get('function')
        if not isinstance(function,dict) or set(function) != {'name','arguments'}:
            raise ValueError('tool call requires name and arguments')
        name=_name(function['name'])
        arguments=_json(function['arguments'])
        if not isinstance(arguments,dict) or len(arguments)>MAX_PARAMETERS:
            raise ValueError('tool arguments require at most 256 object fields')
        if index:append('\n')
        append(f'<{DSML}invoke name="{name}">\n')
        for key,value in arguments.items():
            key=_name(key)
            string=isinstance(value,str)
            body=_value(value if string else _dump(value))
            append(f'<{DSML}parameter name="{key}" string="{str(string).lower()}">{body}{PARAM_END}\n')
        # The canonical encoder has a blank arguments line for zero parameters.
        if not arguments:append('\n')
        append(INVOKE_END)
    append('\n'+END)
    return ''.join(parts)


def parse_tool_calls(text):
    """Parse a complete canonical block atomically; reject partial calls."""
    _text(text)
    if not text.startswith(START):raise ValueError('missing DSML tool block')
    cursor=len(START)
    calls=[]
    while True:
        if len(calls)>=MAX_CALLS:raise ValueError('too many DSML calls')
        match=INVOKE.match(text,cursor)
        if match is None:raise ValueError('malformed DSML invocation')
        name=_name(match[1]);cursor=match.end()
        arguments={}
        if text.startswith('\n'+INVOKE_END,cursor):cursor+=1
        while not text.startswith(INVOKE_END,cursor):
            if len(arguments)>=MAX_PARAMETERS:raise ValueError('too many DSML parameters')
            match=PARAM.match(text,cursor)
            if match is None:raise ValueError('malformed DSML parameter')
            key=_name(match[1])
            if key in arguments:raise ValueError('duplicate DSML parameter')
            stop=text.find(PARAM_END,match.end())
            if stop<0:raise ValueError('unterminated DSML parameter')
            value=_value(text[match.end():stop])
            arguments[key]=value if match[2]=='true' else _json(value)
            cursor=stop+len(PARAM_END)
            if not text.startswith('\n',cursor):raise ValueError('missing DSML parameter newline')
            cursor+=1
        cursor+=len(INVOKE_END)
        calls.append({'type':'function','function':{'name':name,'arguments':_dump(arguments)}})
        if not text.startswith('\n',cursor):raise ValueError('missing DSML invocation newline')
        cursor+=1
        if text.startswith(END,cursor):
            if cursor+len(END)!=len(text):raise ValueError('unexpected text after DSML block')
            return calls
