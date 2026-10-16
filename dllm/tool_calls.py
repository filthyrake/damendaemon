"""Function-tool request validation and incremental JSON-envelope extraction.

This module describes calls; it never imports or executes a supplied function.
Only complete, declared, well-formed calls become structured output. Other
model output is preserved as text, including incomplete calls at the token cap.
"""
import copy
import json
import re
import uuid

_NAME=re.compile(r'[A-Za-z0-9_-]{1,64}\Z')
_ID=re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
_OPEN='<tool_call>'
_CLOSE='</tool_call>'


def _name(value):
    return isinstance(value,str) and _NAME.fullmatch(value) is not None


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def _invalid_constant(value):
    raise ValueError('non-finite JSON number')


def _json_object(value):
    try:result=json.loads(value,object_pairs_hook=_pairs,parse_constant=_invalid_constant)
    except RecursionError as error:raise ValueError('tool JSON nesting is too deep') from error
    if not isinstance(result,dict):raise ValueError('tool arguments must be a JSON object')
    # Reject overflowed numbers and decoded lone surrogates before publishing
    # arguments or replaying history. ASCII escaping would hide invalid Unicode.
    json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8')
    return result


def validate_tools(tools):
    if tools is None:return None
    if not isinstance(tools,list) or len(tools)>64:raise ValueError('tools must be a list of at most 64 functions')
    names=set()
    for tool in tools:
        if not isinstance(tool,dict) or set(tool)!={'type','function'} or tool['type']!='function':
            raise ValueError('only function tools are supported')
        function=tool['function']
        if not isinstance(function,dict) or set(function)-{'name','description','parameters','strict'}:
            raise ValueError('invalid function definition')
        name=function.get('name')
        if not _name(name) or name in names:raise ValueError('function names must be valid and unique')
        names.add(name)
        if 'description' in function and not isinstance(function['description'],str):
            raise ValueError('function description must be text')
        schema=function.get('parameters',{'type':'object'})
        if not isinstance(schema,dict) or schema.get('type','object')!='object':
            raise ValueError('function parameters must describe an object')
        if 'strict' in function and type(function['strict']) is not bool:
            raise ValueError('strict must be boolean')
    try:
        if len(json.dumps(tools,allow_nan=False))>256*1024:raise ValueError('tool definitions are too large')
        return copy.deepcopy(tools)
    except RecursionError as error:
        raise ValueError('tool definition nesting is too deep') from error


def prepare_tools(tools,choice=None):
    """Select prompt definitions and an optional single-call JSON grammar."""
    tools=validate_tools(tools)
    if choice is None:choice='auto' if tools else 'none'
    if choice=='none':return None,None
    if not tools:raise ValueError('tool choice requires function definitions')
    if choice=='auto':
        if any(t['function'].get('strict',False) for t in tools):
            raise ValueError('strict tools require required or named tool_choice')
        return tools,None
    if isinstance(choice,dict):
        function=choice.get('function')
        if (set(choice)!={'type','function'} or choice['type']!='function'
                or not isinstance(function,dict) or set(function)!={'name'} or not _name(function['name'])):
            raise ValueError('invalid named tool choice')
        tools=[t for t in tools if t['function']['name']==function['name']]
        if not tools:raise ValueError('named tool choice is not declared')
    elif choice!='required':raise ValueError('unsupported tool_choice')
    branches=[]
    for tool in tools:
        function=tool['function']
        parameters=copy.deepcopy(function.get('parameters',{'type':'object'}))
        parameters.setdefault('type','object')
        branches.append({'type':'object','properties':{
            'name':{'type':'string','const':function['name']},'arguments':parameters},
            'required':['name','arguments'],'additionalProperties':False})
    from .structured import response_schema
    schema=branches[0] if len(branches)==1 else {'anyOf':branches}
    encoded=response_schema({'type':'json_schema','json_schema':{'name':'tool_call','schema':schema}})
    return tools,encoded


def select_tools(tools,choice=None):
    return prepare_tools(tools,choice)[0]


def validate_messages(messages):
    if not isinstance(messages,list) or not messages:raise ValueError('messages must be a nonempty list')
    pending={};seen=set()
    for message in messages:
        if not isinstance(message,dict):raise ValueError('invalid chat message')
        role=message.get('role');content=message.get('content')
        if pending and role!='tool':raise ValueError('tool calls need results before the next message')
        if role=='tool':
            if set(message)-{'role','content','tool_call_id'} or not isinstance(content,str):
                raise ValueError('tool results require text and a call ID')
            call_id=message.get('tool_call_id')
            if not isinstance(call_id,str) or call_id not in pending:raise ValueError('unknown or duplicate tool result ID')
            del pending[call_id]
        elif role=='assistant' and message.get('tool_calls') not in (None,[]):
            calls=message['tool_calls']
            if (set(message)-{'role','content','tool_calls','reasoning_content'} or content is not None and not isinstance(content,str)
                    or not isinstance(calls,list) or not 1<=len(calls)<=64):
                raise ValueError('invalid assistant tool calls')
            if message.get('reasoning_content') is not None and not isinstance(message['reasoning_content'],str):
                raise ValueError('assistant reasoning history must be text or null')
            for call in calls:
                if not isinstance(call,dict) or set(call)!={'id','type','function'} or call['type']!='function':
                    raise ValueError('invalid historical tool call')
                call_id=call['id'];function=call['function']
                if not isinstance(call_id,str) or not _ID.fullmatch(call_id) or call_id in seen:
                    raise ValueError('tool call IDs must be valid and unique')
                if (not isinstance(function,dict) or set(function)!={'name','arguments'}
                        or not _name(function['name']) or not isinstance(function['arguments'],str)):
                    raise ValueError('invalid historical function')
                _json_object(function['arguments'])
                seen.add(call_id);pending[call_id]=function['name']
        elif role in ('system','user','assistant'):
            allowed={'role','content'}|({'tool_calls','reasoning_content'} if role=='assistant' else set())
            if set(message)-allowed or not isinstance(content,str):raise ValueError('chat content must be text')
            if message.get('reasoning_content') is not None and not isinstance(message['reasoning_content'],str):
                raise ValueError('assistant reasoning history must be text or null')
        else:raise ValueError('unsupported message role')
    if pending:raise ValueError('tool calls need results before generation')
    return copy.deepcopy(messages)


def template_messages(messages):
    """Published Qwen2 templates expect parsed argument objects, not JSON strings."""
    result=validate_messages(messages)
    for message in result:
        for call in message.get('tool_calls') or []:
            call['function']={**call['function'],'arguments':_json_object(call['function']['arguments'])}
    return result


class ToolCallStream:
    def __init__(self,tools):
        self.names={tool['function']['name'] for tool in tools}
        self.buffer='';self.calls=[];self.thinking=False;self.in_call=False
        self.cursor=0;self.depth=0;self.quoted=False;self.escaped=False;self.json_started=False
        self.json_end=None;self.malformed=False

    def _begin(self):
        self.in_call=True;self.cursor=len(_OPEN);self.depth=0;self.quoted=False;self.escaped=False
        self.json_started=False;self.json_end=None;self.malformed=False

    def _envelope(self,final):
        # Scan each newly received character once. Parse JSON only after its
        # balanced object ends, so closing tags inside argument strings are safe.
        while not self.malformed and self.json_end is None and self.cursor<len(self.buffer):
            char=self.buffer[self.cursor];self.cursor+=1
            if not self.json_started:
                if char.isspace():continue
                if char!='{':self.malformed=True;break
                self.json_started=True;self.depth=1;continue
            if self.quoted:
                if self.escaped:self.escaped=False
                elif char=='\\':self.escaped=True
                elif char=='"':self.quoted=False
            elif char=='"':self.quoted=True
            elif char in '[{':self.depth+=1
            elif char in ']}':
                self.depth-=1
                if self.depth==0:self.json_end=self.cursor
        if self.json_end is not None and not self.malformed:
            end=self.json_end
            while end<len(self.buffer) and self.buffer[end].isspace():end+=1
            tail=self.buffer[end:]
            if tail.startswith(_CLOSE):
                consumed=end+len(_CLOSE)
                try:
                    call=_json_object(self.buffer[len(_OPEN):self.json_end])
                    if (len(self.calls)>=64 or set(call)!={'name','arguments'} or not isinstance(call['name'],str)
                            or call['name'] not in self.names or not isinstance(call['arguments'],dict)):
                        raise ValueError('invalid generated call')
                    arguments=json.dumps(call['arguments'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
                except (ValueError,TypeError,RecursionError):
                    return consumed,{'content':self.buffer[:consumed]}
                item={'id':'call_'+uuid.uuid4().hex,'type':'function',
                    'function':{'name':call['name'],'arguments':arguments}}
                index=len(self.calls);self.calls.append(item)
                return consumed,{'tool_calls':[{'index':index,**item}]}
            if not _CLOSE.startswith(tail):self.malformed=True
        if self.malformed:
            end=self.buffer.find(_CLOSE,len(_OPEN))
            if end>=0:
                end+=len(_CLOSE)
                return end,{'content':self.buffer[:end]}
        if final:return len(self.buffer),{'content':self.buffer}
        return None

    def finish(self,reason):
        return self.feed(final=True)

    def feed(self,text='',final=False):
        self.buffer+=text;deltas=[]
        while self.buffer:
            if self.in_call:
                result=self._envelope(final)
                if result is None:break
                consumed,delta=result;deltas.append(delta);self.buffer=self.buffer[consumed:];self.in_call=False
                continue
            markers=('</think>',) if self.thinking else (_OPEN,'<think>','</think>')
            matches=[(self.buffer.find(marker),marker) for marker in markers if marker in self.buffer]
            if matches:
                offset,marker=min(matches)
                if offset:
                    deltas.append({'content':self.buffer[:offset]});self.buffer=self.buffer[offset:]
                if marker==_OPEN:self._begin();continue
                deltas.append({'content':marker});self.buffer=self.buffer[len(marker):]
                self.thinking=marker=='<think>'
                continue
            keep=0
            if not final:
                for marker in markers:
                    for size in range(1,min(len(marker),len(self.buffer)+1)):
                        if self.buffer.endswith(marker[:size]):keep=max(keep,size)
            emit=len(self.buffer)-keep
            if emit:deltas.append({'content':self.buffer[:emit]});self.buffer=self.buffer[emit:]
            break
        return deltas


class JsonToolCallStream:
    """Expose a forced call only after a complete, independently checked value."""
    def __init__(self,schema):
        self.schema=json.loads(schema);self.parts=[];self.calls=[]

    def feed(self,text):
        self.parts.append(text)
        return []

    def finish(self,reason):
        from .structured import matches_schema
        text=''.join(self.parts);self.parts.clear()
        try:
            value=_json_object(text)
            if not matches_schema(value,self.schema):raise ValueError('call violates tool schema')
        except (ValueError,TypeError,RecursionError) as error:
            if reason=='length':return [{'content':text}] if text else []
            raise RuntimeError('incomplete or invalid constrained tool call') from error
        call={'id':'call_'+uuid.uuid4().hex,'type':'function','function':{
            'name':value['name'],'arguments':json.dumps(value['arguments'],ensure_ascii=False,
                separators=(',',':'),allow_nan=False)}}
        self.calls.append(call)
        return [{'tool_calls':[{'index':0,**call}]}]
