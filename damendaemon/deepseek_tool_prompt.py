"""V4 tool histories, independently validated and normalized to text turns.

TOOLS_TEMPLATE is from DeepSeek's MIT-licensed encoding/encoding_dsv4.py,
revision 60d8d70770c6776ff598c94bb586a859a38244f1.
Copyright (c) 2023 DeepSeek. See docs/licenses/deepseek-MIT.txt.
"""
import json

from .deepseek_tools import render_tool_calls, _name, _json
from .structured import response_schema, matches_schema
from .deepseek_text import MAX_TEXT_BYTES, THINK, END_THINK

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def prepare_tools(tools):
    """Render declarations and own a detached, validated name/schema contract."""
    if not isinstance(tools,list) or not 1 <= len(tools) <= 64:
        raise ValueError('V4 tools require 1..64 function schemas')
    contract={}
    schemas=[]
    size=0
    for tool in tools:
        if not isinstance(tool,dict) or set(tool) != {'type','function'} or tool['type'] != 'function':
            raise ValueError('unsupported V4 tool schema')
        function=tool['function']
        if not isinstance(function,dict) or set(function)-{'name','description','parameters','strict'}:
            raise ValueError('unsupported V4 function fields')
        name=_name(function.get('name'))
        if name in contract:raise ValueError('duplicate V4 tool name')
        if ('description' in function and not isinstance(function['description'],str)
                or 'parameters' in function and not isinstance(function['parameters'],dict)
                or 'strict' in function and type(function['strict']) is not bool):
            raise ValueError('invalid V4 function schema')
        try:encoded=json.dumps(function,ensure_ascii=False,allow_nan=False)
        except (TypeError,ValueError,RecursionError):raise ValueError('invalid V4 function JSON') from None
        _json(encoded) # finite JSON and bounded Unicode, including nested schema values
        size+=len(encoded.encode('utf-8'))
        if size>MAX_TEXT_BYTES:raise ValueError('V4 schemas exceed text byte limit')
        parameters=None
        if function.get('strict') is True:
            parameters=function.get('parameters', {'type':'object','additionalProperties':False})
            if parameters.get('type') != 'object':
                raise ValueError('strict V4 parameters require an object schema')
            parameters=json.loads(response_schema({'type':'json_schema','json_schema':{
                'name':'tool_parameters','schema':parameters}}))
        contract[name]=parameters
        schemas.append(encoded)
    prompt=TOOLS_TEMPLATE.format(dsml_token='｜DSML｜',thinking_start_token=THINK,
                                 thinking_end_token=END_THINK,tool_schemas='\n'.join(schemas))
    if len(prompt.encode('utf-8'))>MAX_TEXT_BYTES:raise ValueError('V4 tool prompt exceeds text byte limit')
    return prompt,contract


def tool_schema_prompt(tools):
    prompt,contract=prepare_tools(tools)
    return prompt,set(contract)


def validate_tool_arguments(calls,contract):
    """Check an entire parsed/renderable call block before it is published."""
    for call in calls:
        function=call['function']
        name=function['name']
        if name not in contract:
            raise ValueError('undeclared V4 tool')
        schema=contract[name]
        if schema is not None and not matches_schema(_json(function['arguments']),schema):
            raise ValueError('V4 tool arguments violate strict parameter schema')


def render_tool_chat(messages,tools,*,enable_thinking=False,drop_thinking=True,reasoning_effort=None):
    from .deepseek_text import render_chat
    if type(drop_thinking) is not bool:raise ValueError('thinking retention must be boolean')
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:
        raise ValueError('V4 tool chat requires 1..4096 messages')
    schema,contract=prepare_tools(tools)
    prepared=[]
    index=0
    consumed=len(schema.encode('utf-8'))
    def append(message):
        nonlocal consumed
        for field in ('content','reasoning_content'):
            value=message.get(field,'')
            if not isinstance(value,str):raise ValueError('V4 tool history requires text')
            consumed+=len(value.encode('utf-8'))
        if consumed>MAX_TEXT_BYTES:raise ValueError('V4 tool history exceeds text byte limit')
        prepared.append(message)
    # Inject schemas once in the leading system turn. Tools always retain reasoning
    # in the pinned format, even when drop_thinking has its ordinary-text default.
    if isinstance(messages[0],dict) and messages[0].get('role')=='system':
        system=messages[0]
        if set(system)!={'role','content'} or not isinstance(system['content'],str):
            raise ValueError('invalid V4 tool system message')
        append(dict(system));prepared[0]['content']+='\n\n'+schema
        index=1
    else:
        prepared.append({'role':'system','content':'\n\n'+schema})
    while index<len(messages):
        message=messages[index]
        if not isinstance(message,dict):raise ValueError('invalid V4 tool message')
        calls=message.get('tool_calls')
        if calls is None:
            append(dict(message));index+=1
            continue
        if (message.get('role')!='assistant'
                or set(message)-{'role','content','reasoning_content','tool_calls'}):
            raise ValueError('tool calls require an assistant message')
        block=render_tool_calls(calls)
        validate_tool_arguments(calls,contract)
        ids=[]
        for call in calls:
            identity=call.get('id')
            if not isinstance(identity,str) or not 1<=len(identity)<=128 or identity in ids:
                raise ValueError('tool history requires distinct call IDs')
            ids.append(identity)
        content=message.get('content')
        if content is None:content=''
        if not isinstance(content,str):raise ValueError('V4 assistant content must be text')
        normalized={'role':'assistant','content':content+'\n\n'+block}
        if 'reasoning_content' in message:normalized['reasoning_content']=message['reasoning_content']
        append(normalized)
        index+=1
        results={}
        while index<len(messages) and isinstance(messages[index],dict) and messages[index].get('role')=='tool':
            result=messages[index]
            if set(result)!={'role','tool_call_id','content'}:raise ValueError('invalid tool result fields')
            identity=result['tool_call_id']
            if not isinstance(identity,str) or identity not in ids or identity in results:
                raise ValueError('unknown or duplicate tool result ID')
            value=result['content']
            if not isinstance(value,str) or '<tool_result>' in value or '</tool_result>' in value:
                raise ValueError('invalid or ambiguous tool result text')
            consumed+=len(value.encode('utf-8'))
            if consumed>MAX_TEXT_BYTES:raise ValueError('V4 tool history exceeds text byte limit')
            results[identity]=value
            index+=1
        if len(results)!=len(ids):raise ValueError('missing tool results')
        result_text='\n\n'.join('<tool_result>'+results[identity]+'</tool_result>' for identity in ids)
        if index<len(messages) and isinstance(messages[index],dict) and messages[index].get('role')=='user':
            user=messages[index]
            if set(user)!={'role','content'} or not isinstance(user['content'],str):
                raise ValueError('invalid user text after tool results')
            result_text+='\n\n'+user['content'];index+=1
        # Raw result text was counted above; remove that contribution before
        # counting its rendered form, including result delimiters, once.
        consumed-=sum(len(value.encode('utf-8')) for value in results.values())
        append({'role':'user','content':result_text})
    rendered=render_chat(prepared,enable_thinking=enable_thinking,drop_thinking=False,
                         reasoning_effort=reasoning_effort)
    if len(rendered.encode('utf-8'))>MAX_TEXT_BYTES:raise ValueError('V4 rendered tool prompt exceeds byte limit')
    return rendered
