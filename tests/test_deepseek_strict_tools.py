"""Strict schemas are request contracts, not a claim of constrained sampling."""
import copy
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm.deepseek_tool_prompt import prepare_tools, render_tool_chat, validate_tool_arguments
from dllm.deepseek_generation import ReferenceGenerator
from dllm.deepseek_batch_generation import BatchReferenceGenerator
from dllm.deepseek_text import EOS
from dllm.deepseek_tools import render_tool_calls
from tests.test_deepseek_generation import text_fixture, ScriptedModel
from tests.test_deepseek_batch_generation import ScriptedDeepSeek
from tests.test_deepseek_reference_loader import fixture
from scripts import run_deepseek_reference as cli


def tools(schema=None):
    function={'name':'lookup','strict':True}
    if schema is not None:function['parameters']=schema
    return [{'type':'function','function':function}]


def calls(arguments):
    return [{'id':'call_1','type':'function','function':{
        'name':'lookup','arguments':json.dumps(arguments)}}]


SCHEMA={'type':'object','properties':{
    'items':{'type':'array','items':{'type':'object','properties':{
        'count':{'type':'integer'},'mode':{'type':'string','enum':['fast','slow']},
        'flag':{'type':'boolean','const':True},
        'optional':{'anyOf':[{'type':'null'},{'type':'number'}]}},
        'required':['count','mode','flag'],'additionalProperties':False}}},
    'required':['items'],'additionalProperties':False}
GOOD={'items':[{'count':2,'mode':'fast','flag':True,'optional':None}]}


class DeepSeekStrictToolTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        self.messages=[{'role':'user','content':'Hi'}]

    def test_nested_types_required_enum_const_union_and_extra_fields(self):
        _,contract=prepare_tools(tools(SCHEMA))
        validate_tool_arguments(calls(GOOD),contract)
        bad=[{}, {'items':{}}, {'items':[{}]}, {'items':[],'extra':0}]
        for key,value in [('count',True),('mode','wrong'),('flag',False),('optional','bad'),('extra',0)]:
            args=copy.deepcopy(GOOD);args['items'][0][key]=value;bad.append(args)
        for arguments in bad:
            with self.subTest(arguments=arguments),self.assertRaisesRegex(ValueError,'strict parameter schema'):
                validate_tool_arguments(calls(arguments),contract)

    def test_empty_default_and_nonstrict_compatibility(self):
        _,contract=prepare_tools(tools());validate_tool_arguments(calls({}),contract)
        with self.assertRaises(ValueError):validate_tool_arguments(calls({'x':1}),contract)
        for strict in (False,None):
            declarations=tools({'customKeyword':'legacy'})
            if strict is None:del declarations[0]['function']['strict']
            else:declarations[0]['function']['strict']=strict
            _,contract=prepare_tools(declarations)
            validate_tool_arguments(calls({'anything':[True,3]}),contract)

    def test_history_checks_strict_arguments(self):
        def history(arguments):
            return self.messages+[{'role':'assistant','tool_calls':calls(arguments)},
                {'role':'tool','tool_call_id':'call_1','content':'done'}]
        self.assertIn('lookup',render_tool_chat(history(GOOD),tools(SCHEMA)))
        with self.assertRaisesRegex(ValueError,'strict parameter schema'):
            render_tool_chat(history({}),tools(SCHEMA))

    def run_generation(self,arguments,*,batch=False,prefix=''):
        output=prefix+'\n\n'+render_tool_calls(calls(arguments))+EOS
        if batch:
            model=ScriptedDeepSeek(self.text,[output])
            runner=BatchReferenceGenerator(model,self.text,max_context=4096)
            return model,runner.generate_batch([self.messages],max_new_tokens=1024,tools=tools(SCHEMA))
        model=ScriptedModel(self.text,output)
        runner=ReferenceGenerator(model,self.text,max_context=4096)
        return model,runner.generate(self.messages,max_new_tokens=1024,tools=tools(SCHEMA))

    def test_single_and_batch_valid_and_invalid_output_owned_cleanup(self):
        for batch in (False,True):
            model,stream=self.run_generation(GOOD,batch=batch)
            events=list(stream)
            self.assertEqual(events[-1]['finish_reason'],'tool_calls')
            self.assertEqual(json.loads(next(e for e in events if 'tool_calls' in e)['tool_calls'][0]['function']['arguments']),GOOD)
            self.assertTrue(model.states[-1].closed)
            model,stream=self.run_generation({},batch=batch);events=[]
            with self.assertRaisesRegex(ValueError,'strict parameter schema'):
                for event in stream:events.append(event)
            self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events))
            self.assertTrue(model.states[-1].closed)

    def test_entire_call_block_is_atomic(self):
        block=calls(GOOD)+calls({});block[1]['id']='call_2'
        output='\n\n'+render_tool_calls(block)+EOS
        for batch in (False,True):
            model=ScriptedDeepSeek(self.text,[output]) if batch else ScriptedModel(self.text,output)
            runner=(BatchReferenceGenerator if batch else ReferenceGenerator)(model,self.text,max_context=4096)
            stream=(runner.generate_batch([self.messages],max_new_tokens=1024,tools=tools(SCHEMA)) if batch
                    else runner.generate(self.messages,max_new_tokens=1024,tools=tools(SCHEMA)))
            events=[]
            with self.assertRaisesRegex(ValueError,'strict parameter schema'):
                for event in stream:events.append(event)
            self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events))
            self.assertTrue(model.states[-1].closed)

    def test_schema_snapshot_cannot_be_relaxed_or_tightened_midstream(self):
        for batch in (False,True):
            for valid in (False,True):
                declarations=tools(copy.deepcopy(SCHEMA));arguments=GOOD if valid else {}
                output='lead\n\n'+render_tool_calls(calls(arguments))+EOS
                model=ScriptedDeepSeek(self.text,[output]) if batch else ScriptedModel(self.text,output)
                runner=(BatchReferenceGenerator if batch else ReferenceGenerator)(model,self.text,max_context=4096)
                stream=(runner.generate_batch([self.messages],max_new_tokens=1024,tools=declarations) if batch
                        else runner.generate(self.messages,max_new_tokens=1024,tools=declarations))
                next(stream)
                declarations[0]['function']['parameters'].clear()
                declarations[0]['function']['strict']=False
                if valid:self.assertEqual(list(stream)[-1]['finish_reason'],'tool_calls')
                else:
                    with self.assertRaisesRegex(ValueError,'strict parameter schema'):list(stream)
                self.assertTrue(model.states[-1].closed)

    def test_unsupported_schema_preflight_before_prefill_and_cli_payload(self):
        fixture(self.path,vocab_size=self.text.vocab_size)
        tool_file=self.path/'tools.json';message_file=self.path/'messages.json'
        invalid=[{'type':'array','items':{'type':'string'}},
                 {'type':'object','patternProperties':{}},
                 {'type':'object','description':'x'*16384}]
        deep={'type':'string'}
        for _ in range(18):deep={'type':'array','items':deep}
        invalid.append({'type':'object','properties':{'x':deep}})
        invalid.append({'type':'object','properties':{str(i):{'type':'string'} for i in range(257)}})
        for schema in invalid:
            for batch in (False,True):
                declarations=tools(schema);tool_file.write_text(json.dumps(declarations))
                message_file.write_text(json.dumps([self.messages] if batch else self.messages))
                argv=[str(self.path),'--batch-messages' if batch else '--messages',str(message_file),'--tools',str(tool_file)]
                with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:cli.main(argv)
                    self.assertEqual(error.exception.code,2)
                model=ScriptedDeepSeek(self.text,['answer'+EOS]) if batch else ScriptedModel(self.text,'answer'+EOS)
                runner=(BatchReferenceGenerator if batch else ReferenceGenerator)(model,self.text,max_context=4096)
                with self.assertRaises(ValueError):
                    list(runner.generate_batch([self.messages],tools=declarations) if batch else runner.generate(self.messages,tools=declarations))
                self.assertFalse(model.states)

    def test_invalid_history_rejected_before_cli_payload(self):
        fixture(self.path,vocab_size=self.text.vocab_size)
        declarations=self.path/'tools.json';declarations.write_text(json.dumps(tools(SCHEMA)))
        history=self.messages+[{'role':'assistant','tool_calls':calls({})},
            {'role':'tool','tool_call_id':'call_1','content':'done'}]
        for batch in (False,True):
            messages=self.path/'history.json';messages.write_text(json.dumps([history] if batch else history))
            argv=[str(self.path),'--batch-messages' if batch else '--messages',str(messages),'--tools',str(declarations)]
            with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:cli.main(argv)
                self.assertEqual(error.exception.code,2)
                self.assertIn('strict parameter schema',stderr.getvalue())

    def test_strict_declaration_still_allows_ordinary_text(self):
        model=ScriptedDeepSeek(self.text,['answer'+EOS,'other'+EOS])
        runner=BatchReferenceGenerator(model,self.text,max_context=4096)
        events=list(runner.generate_batch([self.messages]*2,tools=tools(SCHEMA)))
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['stop','stop'])
        self.assertFalse(any('tool_calls' in e for e in events))
        self.assertTrue(model.states[-1].closed)
