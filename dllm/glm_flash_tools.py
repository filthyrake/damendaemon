"""Schema-bound Flash tool-call codec; never executes tools.

Wire serialization follows the MIT-licensed GLM-5.3-Flash chat_template.jinja
at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a. See
docs/licenses/glm53-flash-MIT.txt. Strings are raw; other values are JSON.
Declared top-level argument types remove that otherwise ambiguous distinction.
"""
import json
import re

from .checkpoint import _unique_fields
from .glm_text import MAX_TEXT_BYTES, CONTROLS
from .structured import response_schema, matches_schema

OPEN='<tool_call>'
CLOSE='</tool_call>'
KEY='<arg_key>'
END_KEY='</arg_key>'
VALUE='<arg_value>'
END_VALUE='</arg_value>'
RESERVED=CONTROLS+('<|','<tool_call','</tool_call','<tool_response','</tool_response','<arg_','</arg_')


def _text(value, limit=MAX_TEXT_BYTES):
    if not isinstance(value,str):raise ValueError('Flash tool text must be a string')
    try:size=len(value.encode('utf-8'))
    except UnicodeError:raise ValueError('invalid Flash tool Unicode') from None
    if size>limit:raise ValueError('Flash tool text exceeds byte limit')
    return value


def _name(value):
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',value) is None:
        raise ValueError('Flash tool/argument names require 1..128 ASCII identifier characters')
    return value


def _dump(value):
    try:return _text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    except (ValueError,TypeError,RecursionError):raise ValueError('invalid bounded Flash tool JSON') from None


def _json(value):
    def invalid(value):raise ValueError('nonfinite Flash tool JSON')
    try:
        result=json.loads(_text(value),object_pairs_hook=_unique_fields,parse_constant=invalid)
        _dump(result)  # Reject exponent overflow, nested nonfinite values and oversized normalization.
        return result
    except (ValueError,RecursionError):raise ValueError('Flash tool arguments require finite duplicate-free JSON') from None


def _literal(value):
    _text(value)
    if any(marker in value for marker in RESERVED):raise ValueError('Flash tool argument contains a reserved delimiter/control')
    return value


class FlashTools:
    """Immutable declaration snapshot for a deliberately bounded schema subset."""
    def __init__(self,tools):
        # Copy through finite JSON so subsequent caller mutations cannot change
        # the interpretation of an already accepted wire-format contract.
        tools=_json(_dump(tools))
        if not isinstance(tools,list) or not 1<=len(tools)<=64:
            raise ValueError('Flash tools require 1..64 function declarations')
        schemas={}
        for tool in tools:
            if not isinstance(tool,dict) or set(tool)!={'type','function'} or tool['type']!='function':
                raise ValueError('unsupported Flash tool declaration')
            function=tool['function']
            if not isinstance(function,dict) or set(function)-{'name','description','parameters','strict'}:
                raise ValueError('unsupported Flash function fields')
            name=_name(function.get('name'))
            if name in schemas:raise ValueError('duplicate Flash function name')
            if 'description' in function:_text(function['description'],16384)
            if 'strict' in function and type(function['strict']) is not bool:raise ValueError('invalid Flash strict flag')
            schema=function.get('parameters',{'type':'object','properties':{},'additionalProperties':False})
            encoded=response_schema(dict(type='json_schema',json_schema=dict(name='flash_tool',schema=schema)))
            schema=_json(encoded)
            if schema.get('type')!='object' or schema.get('additionalProperties',False) is not False:
                raise ValueError('Flash tool parameters require a closed object schema')
            for key,child in schema.get('properties',{}).items():
                _name(key)
                if 'type' not in child:raise ValueError('Flash argument wire types must be unambiguous')
            schemas[name]=schema
        self._schemas=schemas

    def _schema(self,name):
        _name(name)
        if name not in self._schemas:raise ValueError('undeclared Flash tool name')
        return self._schemas[name]

    def render(self,calls):
        if not isinstance(calls,list) or not 1<=len(calls)<=64:raise ValueError('Flash requires 1..64 tool calls')
        parts=[];size=0
        def append(value):
            nonlocal size
            size+=len(value.encode('utf-8'))
            if size>MAX_TEXT_BYTES:raise ValueError('Flash tool calls exceed byte limit')
            parts.append(value)
        for call in calls:
            if (not isinstance(call,dict) or set(call)-{'type','function','id'} or call.get('type')!='function'
                    or not isinstance(call.get('function'),dict) or set(call['function'])!={'name','arguments'}):
                raise ValueError('invalid Flash tool call')
            if 'id' in call:_text(call['id'],128)
            function=call['function'];name=function['name'];schema=self._schema(name)
            arguments=_json(function['arguments'])
            if not matches_schema(arguments,schema):raise ValueError('Flash tool arguments do not match declared schema')
            append(OPEN+name)
            for key,value in arguments.items():
                body=value if schema.get('properties',{})[key]['type']=='string' else _dump(value)
                append(KEY+key+END_KEY+VALUE+_literal(body)+END_VALUE)
            append(CLOSE)
        return ''.join(parts)

    def parse(self,text):
        text=_text(text);cursor=0;calls=[]
        while cursor<len(text):
            while cursor<len(text) and text[cursor] in ' \t\r\n':cursor+=1
            if cursor==len(text):break
            if len(calls)>=64 or not text.startswith(OPEN,cursor):raise ValueError('invalid Flash tool call boundary')
            cursor+=len(OPEN)
            end=text.find('<',cursor)
            if end<0:raise ValueError('incomplete Flash tool call')
            name=text[cursor:end];schema=self._schema(name);cursor=end;arguments={}
            while not text.startswith(CLOSE,cursor):
                if not text.startswith(KEY,cursor):raise ValueError('missing Flash argument key')
                start=cursor+len(KEY);end=text.find(END_KEY,start)
                if end<0:raise ValueError('incomplete Flash argument key')
                key=_name(text[start:end]);cursor=end+len(END_KEY)
                if key in arguments or key not in schema.get('properties',{}):raise ValueError('duplicate or undeclared Flash argument')
                if not text.startswith(VALUE,cursor):raise ValueError('missing Flash argument value')
                start=cursor+len(VALUE);end=text.find(END_VALUE,start)
                if end<0:raise ValueError('incomplete Flash argument value')
                raw=_literal(text[start:end]);cursor=end+len(END_VALUE)
                arguments[key]=raw if schema['properties'][key]['type']=='string' else _json(raw)
            cursor+=len(CLOSE)
            if not matches_schema(arguments,schema):raise ValueError('Flash tool arguments do not match declared schema')
            calls.append(dict(id='call_'+str(len(calls)),type='function',function=dict(name=name,arguments=_dump(arguments))))
        if not calls:raise ValueError('missing Flash tool calls')
        return calls
