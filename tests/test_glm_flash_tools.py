import json
from pathlib import Path
import unittest

from dllm.glm_flash_tools import FlashTools, MAX_TEXT_BYTES


def tools(properties=None, required=None):
    return [dict(type='function',function=dict(name='f',parameters=dict(type='object',
        properties=properties or {},required=required or [],additionalProperties=False)))]


def call(arguments,name='f'):
    return dict(type='function',function=dict(name=name,arguments=json.dumps(arguments,ensure_ascii=False)))


class FlashToolsTests(unittest.TestCase):
    def test_pinned_template_serialization_and_typed_roundtrip(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/glm53_flash_tool_codec_contract.json').read_text())
        self.assertEqual(fixture['revision'],'eb9eb208eb0d988989d07a6a12d0fdeb5f52574a')
        for case in fixture['cases']:
            codec=FlashTools(case['tools'])
            self.assertEqual(codec.render(case['calls']),case['wire'])
            actual=codec.parse(case['wire'])
            self.assertEqual([c['id'] for c in actual],['call_'+str(i) for i in range(len(actual))])
            for expected,parsed in zip(case['calls'],actual):
                self.assertEqual(expected['function']['name'],parsed['function']['name'])
                self.assertEqual(json.loads(expected['function']['arguments']),json.loads(parsed['function']['arguments']))

    def test_strings_that_look_like_json_are_never_guessed(self):
        codec=FlashTools(tools({'value':{'type':'string'}},['value']))
        for value in ('123','false','null','NaN','1e999','{"a":2}','[1,2]','', '  x\n '):
            result=codec.parse(codec.render([call({'value':value})]))
            self.assertEqual(json.loads(result[0]['function']['arguments']),{'value':value})

    def test_schema_types_required_fields_and_enum_enforced(self):
        codec=FlashTools(tools({'count':{'type':'integer','enum':[1,2]},'flag':{'type':'boolean'},
                               'name':{'type':'string','const':'x'}},['count']))
        for arguments in ({},{'count':True},{'count':3},{'count':1,'flag':1},{'count':1,'name':'y'},{'count':1,'extra':0}):
            with self.subTest(arguments=arguments),self.assertRaises(ValueError):codec.render([call(arguments)])
        for raw in ('true','3','1.0','null','NaN','1e999'):
            with self.assertRaises(ValueError):codec.parse('<tool_call>f<arg_key>count</arg_key><arg_value>'+raw+'</arg_value></tool_call>')
        self.assertEqual(json.loads(codec.parse(codec.render([call({'count':2,'flag':False})]))[0]['function']['arguments']),{'count':2,'flag':False})

    def test_ambiguous_or_open_schemas_rejected_and_declarations_snapshotted(self):
        variants=[tools({'x':{'anyOf':[{'type':'string'},{'type':'integer'}]}}),tools({'bad<key':{'type':'string'}}),
                  tools({'x':{'type':'string','pattern':'x'}})]
        opened=tools();opened[0]['function']['parameters']['additionalProperties']=True;variants.append(opened)
        variants.extend(([],tools()*2,[dict(type='function',function=dict(name='f',defer_loading=True))]))
        for declaration in variants:
            with self.assertRaises(ValueError):FlashTools(declaration)
        declaration=tools({'x':{'type':'string'}});codec=FlashTools(declaration)
        declaration[0]['function']['parameters']['properties']['x']['type']='integer'
        self.assertEqual(json.loads(codec.parse(codec.render([call({'x':'123'})]))[0]['function']['arguments']),{'x':'123'})

    def test_duplicate_delimiters_controls_and_partial_calls_rejected_atomically(self):
        codec=FlashTools(tools({'x':{'type':'string'},'obj':{'type':'object','additionalProperties':True}}))
        valid=codec.render([call({'x':'ok'})])
        for bad in ('',valid[:-1],valid+'garbage',valid+'<tool_call>unknown</tool_call>',
                    valid.replace('</tool_call>','<arg_key>x</arg_key><arg_value>again</arg_value></tool_call>'),
                    valid.replace('ok','<|image|>'),valid.replace('ok','<tool_call>f'),
                    '<tool_call>f<arg_key>obj</arg_key><arg_value>{"a":1,"a":2}</arg_value></tool_call>'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):codec.parse(bad)
        for value in ('<arg_value>','</think>','<|observation|>','<tool_response>'):
            with self.assertRaises(ValueError):codec.render([call({'x':value})])
        duplicate=call({});duplicate['function']['arguments']='{"x":"a","x":"b"}'
        with self.assertRaises(ValueError):codec.render([duplicate])
        self.assertEqual(len(codec.parse(' \n'+valid+'\t'+valid+'\r')),2)

    def test_escaped_nested_controls_fail_before_atomic_publication(self):
        from dllm.glm_flash_tools import RESERVED
        from dllm.glm_flash_tool_output import FlashToolOutput
        from dllm.glm_text import END_THINK,EOS
        codec=FlashTools(tools({'obj':{'type':'object','additionalProperties':True}}))
        valid=codec.render([call({'obj':{'x':'ordinary é🙂'}})])
        for marker in RESERVED:
            escaped=''.join('\\u%04x' % ord(c) for c in marker)
            for raw in ('{"nested":["'+escaped+'"]}','{"'+escaped+'":"value"}'):
                bad='<tool_call>f<arg_key>obj</arg_key><arg_value>'+raw+'</arg_value></tool_call>'
                with self.subTest(marker=marker,raw=raw):
                    with self.assertRaises(ValueError):codec.parse(valid+bad)
                    parser=FlashToolOutput(codec)
                    self.assertEqual(parser.feed(END_THINK+valid+bad+EOS),[])
                    with self.assertRaises(ValueError):parser.finish()
                    self.assertTrue(parser.failed)
                    self.assertTrue(parser.buffer.closed)
        self.assertEqual(codec.render(codec.parse(valid)),valid)

    def test_byte_count_call_count_and_nested_nonfinite_bounds(self):
        codec=FlashTools(tools({'x':{'type':'string'},'obj':{'type':'object','additionalProperties':True}}))
        for calls in ([],[call({})]*65,[call({'x':'🦙'*(MAX_TEXT_BYTES//4)})]):
            with self.assertRaises(ValueError):codec.render(calls)
        with self.assertRaises(ValueError):codec.parse('<tool_call>f</tool_call>'*65)
        with self.assertRaises(ValueError):codec.parse('x'*(MAX_TEXT_BYTES+1))
        with self.assertRaises(ValueError):codec.render([call({'obj':{'x':float('inf')}})])
        with self.assertRaises(ValueError):codec.render([call({'x':'\ud800'})])
        self.assertEqual(codec.parse('<tool_call>f</tool_call>')[0]['function']['arguments'],'{}')
