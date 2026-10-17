import copy
import json
import unittest
from types import SimpleNamespace
from damendaemon.tool_calls import JsonToolCallStream,prepare_tools,ToolCallStream,select_tools,validate_tools,validate_messages,template_messages

TOOLS=[{'type':'function','function':{'name':'weather','description':'Get weather',
    'parameters':{'type':'object','properties':{'city':{'type':'string'}}}}}]


def envelope(arguments,name='weather'):
    return '<tool_call>\n'+json.dumps({'name':name,'arguments':arguments},ensure_ascii=False)+'\n</tool_call>'


def parse(parts):
    parser=ToolCallStream(TOOLS);deltas=[]
    for part in parts:deltas+=parser.feed(part)
    deltas+=parser.feed(final=True)
    return ''.join(d.get('content','') for d in deltas),[c['function'] for c in parser.calls],deltas


class ToolCallTests(unittest.TestCase):
    def test_tool_definitions_preserve_valid_unicode_and_size_limit(self):
        tools = [{'type':'function','function':{'name':'test','description':'Zürich 🦙',
            'parameters':{'type':'object','properties':{'é':{'type':'string','enum':['🦙']}}}}}]
        result = validate_tools(tools)
        self.assertEqual(result,tools)
        self.assertIsNot(result,tools)
        tools[0]['function']['description'] = 'x' * (256*1024)
        with self.assertRaisesRegex(ValueError,'too large'):validate_tools(tools)

    def test_deep_and_cyclic_definitions_are_validation_errors(self):
        nested = json.loads('[' * 600 + '0' + ']' * 600)
        cyclic = {}; cyclic['self'] = cyclic
        for value in (nested, cyclic):
            tools = [{'type': 'function', 'function': {'name': 'test',
                'parameters': {'type': 'object', 'extra': value}}}]
            with self.subTest(cyclic=value is cyclic), self.assertRaises(ValueError):
                validate_tools(tools)

    def test_unqualified_model_template_rejected_before_rendering(self):
        from damendaemon.qwen2 import ModelWeights
        model=SimpleNamespace(template=object(),supports_tool_calls=False)
        with self.assertRaisesRegex(ValueError,'qualified'):
            ModelWeights.chat_tokens(model,[{'role':'user','content':'Weather?'}],tools=TOOLS)

    def test_empty_optional_calls_do_not_turn_text_into_tool_history(self):
        for calls in (None,[]):
            messages=[{'role':'assistant','content':'Hello','tool_calls':calls}]
            self.assertEqual(template_messages(messages),messages)

    def test_all_stream_boundaries_preserve_strings_and_multiple_calls(self):
        args={'city':'Zürich 🦙','literal':'</tool_call> <think> { [ \\"','nested':[{'x':1}]}
        value='Before '+envelope(args)+'\n'+envelope({'city':'Paris'})+' after'
        expected=parse([value])[:2]
        self.assertEqual(expected[0],'Before \n after')
        self.assertEqual(json.loads(expected[1][0]['arguments']),args)
        for cut in range(len(value)+1):
            with self.subTest(cut=cut):self.assertEqual(parse([value[:cut],value[cut:]])[:2],expected)
        self.assertEqual(parse(list(value))[:2],expected)
        deltas=parse(list(value))[2]
        calls=[c for d in deltas for c in d.get('tool_calls',[])]
        self.assertEqual([c['index'] for c in calls],[0,1])
        self.assertNotEqual(calls[0]['id'],calls[1]['id'])

    def test_call_is_emitted_only_after_complete_valid_envelope(self):
        value=envelope({'city':'Paris'});parser=ToolCallStream(TOOLS)
        self.assertEqual(parser.feed(value[:-1]),[])
        self.assertEqual(len(parser.feed(value[-1:])[0]['tool_calls']),1)
        self.assertEqual(parser.feed(final=True),[])

    def test_generated_call_limit_keeps_response_history_replayable(self):
        value=envelope({})
        content,calls,_=parse([value*65])
        self.assertEqual(len(calls),64)
        self.assertEqual(content,value)

    def test_reasoning_markers_do_not_create_tool_calls(self):
        value='<think>Example: '+envelope({'city':'fake'})+'</think>\n'+envelope({'city':'real'})
        content,calls,_=parse(list(value))
        self.assertEqual(content,'<think>Example: '+envelope({'city':'fake'})+'</think>\n')
        self.assertEqual([json.loads(c['arguments']) for c in calls],[{'city':'real'}])
        unclosed='<think>'+envelope({'city':'fake'})
        self.assertEqual(parse(list(unclosed))[:2],(unclosed,[]))

    def test_invalid_and_truncated_output_is_lossless_text(self):
        values=[envelope({},'undeclared'),envelope([]),envelope('{}'),envelope({},name={}),
            '<tool_call>{"name":"weather","arguments":{"x":NaN}}</tool_call>',
            '<tool_call>{"name":"weather","arguments":{"x":1e999}}</tool_call>',
            '<tool_call>{"name":"weather","name":"weather","arguments":{}}</tool_call>',
            '<tool_call>{"name":"weather","arguments":{"x":1,"x":2}}</tool_call>',
            '<tool_call>{bad JSON}</tool_call>', '<tool_call>[]</tool_call>',
            '<tool_call>{"name":"weather","arguments":{}} extra</tool_call>',
            '<tool_call>{"name":"weather","arguments":{"city":"Par',
            envelope({})[:-4],'<tool_ca','hello </think>']
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(parse([value])[:2],(value,[]))
                self.assertEqual(parse(list(value))[:2],(value,[]))

    def test_options_are_copied_and_unsupported_guarantees_rejected(self):
        selected=select_tools(TOOLS)
        selected[0]['function']['name']='other'
        self.assertEqual(TOOLS[0]['function']['name'],'weather')
        self.assertIsNone(select_tools(TOOLS,'none'))
        self.assertIsNone(select_tools(None))
        for tools,choice in ((None,'auto'),(TOOLS,'invalid'),(TOOLS,{'type':'function','function':{'name':'unknown'}})):
            with self.assertRaises(ValueError):select_tools(tools,choice)
        for invalid in ([TOOLS[0],TOOLS[0]], [{'type':'custom','function':{}}],
                [{'type':'function','function':{'name':'bad name'}}],
                [{'type':'function','function':{'name':'weather','strict':0}}],
                [{'type':'function','function':{'name':'weather','parameters':[]}}]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):validate_tools(invalid)

    def test_history_maps_results_and_parses_arguments_without_mutating_input(self):
        history=[{'role':'user','content':'Weather?'},{'role':'assistant','content':None,
            'tool_calls':[{'id':'call_a','type':'function','function':{'name':'weather','arguments':'{"city":"Paris"}'}}]},
            {'role':'tool','tool_call_id':'call_a','content':'Sunny'}]
        original=copy.deepcopy(history)
        normalized=template_messages(history)
        self.assertEqual(normalized[1]['tool_calls'][0]['function']['arguments'],{'city':'Paris'})
        self.assertEqual(history,original)
        cases=[history[:-1],[history[-1]],history+[history[-1]],
            history[:2]+[{'role':'user','content':'Never mind'}],
            history[:2]+[{'role':'tool','tool_call_id':'call_unknown','content':'Sunny'}],
            history+history[1:]]
        invalid=copy.deepcopy(history);invalid[1]['tool_calls'][0]['function']['arguments']='[]';cases.append(invalid)
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_messages(value)


class ForcedToolTests(unittest.TestCase):
    def setUp(self):
        self.tools=[{'type':'function','function':{'name':name,'strict':True,'parameters':{
            'type':'object','properties':{'value':{'type':kind}},'required':['value'],
            'additionalProperties':False}}} for name,kind in [('text','string'),('number','integer')]]

    def test_correlated_union_and_named_choice(self):
        from damendaemon.structured import matches_schema
        selected,encoded=prepare_tools(self.tools,'required');schema=json.loads(encoded)
        self.assertEqual(len(selected),2)
        for value,expected in [({'name':'text','arguments':{'value':'yes'}},True),
                ({'name':'number','arguments':{'value':3}},True),
                ({'name':'number','arguments':{'value':'yes'}},False),
                ({'name':'number','arguments':{'value':True}},False),
                ({'name':'text','arguments':{}},False),
                ({'name':'text','arguments':{'value':'yes','extra':1}},False),
                ({'name':'missing','arguments':{'value':'yes'}},False)]:
            self.assertEqual(matches_schema(value,schema),expected,value)
        selected,encoded=prepare_tools(self.tools,{'type':'function','function':{'name':'number'}})
        self.assertEqual([t['function']['name'] for t in selected],['number'])
        self.assertEqual(json.loads(encoded)['properties']['name']['const'],'number')
        with self.assertRaises(ValueError):prepare_tools(self.tools,'auto')
        self.assertEqual(prepare_tools(self.tools,'none'),(None,None))

    def test_forced_stream_validates_before_emitting(self):
        _,schema=prepare_tools(self.tools,'required')
        value='{"name":"text","arguments":{"value":"Zürich </tool_call>"}}'
        for cut in range(len(value)+1):
            parser=JsonToolCallStream(schema)
            self.assertEqual(parser.feed(value[:cut]),[])
            self.assertEqual(parser.feed(value[cut:]),[])
            result=parser.finish('stop')
            self.assertEqual(json.loads(result[0]['tool_calls'][0]['function']['arguments']),
                {'value':'Zürich </tool_call>'})
            self.assertEqual(len(parser.calls),1)
        for invalid in (value[:-3],'{"name":"number","arguments":{"value":"bad"}}',
                '{"name":"text","name":"text","arguments":{"value":"bad"}}'):
            parser=JsonToolCallStream(schema);parser.feed(invalid)
            with self.assertRaises(RuntimeError):parser.finish('stop')
            parser=JsonToolCallStream(schema);parser.feed(invalid)
            self.assertEqual(parser.finish('length'),[{'content':invalid}])
            self.assertEqual(parser.calls,[])

    def test_invalid_choice_and_schema_rejected(self):
        for choice in ({},[],1,{'type':'function','function':{'name':'text','extra':1}},
                {'type':'function','function':{'name':'unknown'}}):
            with self.subTest(choice=choice),self.assertRaises(ValueError):prepare_tools(self.tools,choice)
        self.tools[0]['function']['parameters']['properties']['value']['pattern']='x'
        with self.assertRaises(ValueError):prepare_tools(self.tools,'required')


class ToolUnicodeTests(unittest.TestCase):
    def test_invalid_decoded_unicode_never_becomes_a_call_or_history(self):
        _, schema = prepare_tools(TOOLS, 'required')
        for arguments in (r'{"city":"\ud800"}', r'{"city":"\udfff"}',
                          r'{"nested":[{"\ud800":"value"}]}',
                          r'{"nested":[{"key":"\udc00"}]}'):
            raw = '{"name":"weather","arguments":' + arguments + '}'
            envelope = '<tool_call>' + raw + '</tool_call>'
            with self.subTest(arguments=arguments):
                for pieces in ([envelope], list(envelope)):
                    content,calls,_ = parse(pieces)
                    self.assertEqual(content,envelope)
                    self.assertEqual(calls,[])
                for reason in ('stop','length'):
                    parser = JsonToolCallStream(schema);parser.feed(raw)
                    if reason == 'length':
                        self.assertEqual(parser.finish(reason),[{'content':raw}])
                    else:
                        with self.assertRaises(RuntimeError):parser.finish(reason)
                    self.assertEqual(parser.calls,[])
                history = [{'role':'assistant','content':None,'tool_calls':[
                    {'id':'call_a','type':'function','function':{'name':'weather','arguments':arguments}}]},
                    {'role':'tool','tool_call_id':'call_a','content':'result'}]
                with self.assertRaises(ValueError):validate_messages(history)

    def test_valid_unicode_and_escaped_pairs_remain_replayable(self):
        _,schema = prepare_tools(TOOLS,'required')
        for arguments in (r'{"city":"\ud83e\udd99"}', '{"city":"Zürich 🦙"}'):
            raw = '{"name":"weather","arguments":' + arguments + '}'
            automatic = ToolCallStream(TOOLS)
            automatic.feed('<tool_call>'+raw+'</tool_call>')
            forced = JsonToolCallStream(schema);forced.feed(raw);forced.finish('stop')
            for parser in (automatic,forced):
                self.assertEqual(len(parser.calls),1)
                function = parser.calls[0]['function']
                function['arguments'].encode('utf-8')
                self.assertEqual(json.loads(function['arguments']),json.loads(arguments))
                history = [{'role':'assistant','content':None,'tool_calls':parser.calls},
                    {'role':'tool','tool_call_id':parser.calls[0]['id'],'content':'result'}]
                self.assertEqual(template_messages(history)[0]['tool_calls'][0]['function']['arguments'],json.loads(arguments))
