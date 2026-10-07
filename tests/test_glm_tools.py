import copy
import io
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm.glm_tool_text import GlmToolText,render_tool_chat
from dllm.glm_tool_output import GlmToolTokenStream
from dllm.glm_tool_generation import GlmToolReferenceGenerator
from dllm.glm_flash_tools import FlashTools
from dllm.glm_text import END_THINK,EOS
from scripts import run_glm_reference as cli
from tests.test_glm_generation import text_fixture,ScriptedModel
from tests.test_glm_batch_generation import ScriptedGlm
from tests.test_glm_reference_loader import fixture

FIXTURE=json.loads((Path(__file__).parent/'fixtures/glm53_tool_prompt_contract.json').read_text())
TOOLS=FIXTURE['tools']
CALLS=[{'type':'function','function':{'name':'lookup','arguments':'{"query":"false","count":2}'}}]


class GlmToolTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);base=text_fixture(self.path)
        self.text=GlmToolText(self.path,vocab_size=base.vocab_size)
        self.messages=[dict(role='user',content='Hi')]
        self.wire=FlashTools(TOOLS).render(CALLS)

    def test_separately_pinned_glm_prompts_and_explicit_adapter(self):
        self.assertEqual(FIXTURE['revision'],'aca966e4e02791568aa6a4ced368624b3d897f42')
        self.assertEqual(FIXTURE['template_sha256'],'3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c')
        for case in FIXTURE['cases']:
            messages=copy.deepcopy(case['messages'])
            self.assertEqual(render_tool_chat(messages,TOOLS,**case['options']),case['prompt'])
            self.assertEqual(self.text.tool_chat_tokens(messages,TOOLS,**case['options']),
                self.text.tokenizer.encode(case['prompt'],add_special_tokens=False).ids)
            self.assertEqual(messages,case['messages'])
        with self.assertRaises(ValueError):self.text.chat_tokens(FIXTURE['cases'][2]['messages'])
        history=copy.deepcopy(FIXTURE['cases'][2]['messages']);history[-1]['tool_call_id']='foreign'
        with self.assertRaises(ValueError):self.text.tool_chat_tokens(history,TOOLS)
        deferred=copy.deepcopy(TOOLS);deferred[0]['function']['defer_loading']=True
        with self.assertRaises(ValueError):self.text.tool_chat_tokens(self.messages,deferred)

    def test_single_tool_output_typed_arguments_and_continuation(self):
        model=ScriptedModel(self.text,'分析'+END_THINK+'checking'+self.wire+EOS)
        runner=GlmToolReferenceGenerator(model,self.text,max_context=4096)
        events=list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=512))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        emitted=next(e['tool_calls'] for e in events if 'tool_calls' in e)
        self.assertEqual(json.loads(emitted[0]['function']['arguments']),{'query':'false','count':2})
        self.assertTrue(model.states[-1].closed)
        emitted[0]['id']='lookup_1'
        history=self.messages+[dict(role='assistant',tool_calls=emitted),dict(role='tool',tool_call_id='lookup_1',content='found')]
        following=ScriptedModel(self.text,END_THINK+'done'+EOS)
        result=list(GlmToolReferenceGenerator(following,self.text,max_context=4096).generate_tools(history,TOOLS,max_new_tokens=512))
        self.assertEqual(''.join(e.get('content','') for e in result),'done')
        self.assertEqual(result[-1]['finish_reason'],'stop')

    def test_mixed_batch_calls_and_independent_finish_reasons(self):
        model=ScriptedGlm(self.text,[END_THINK+self.wire+EOS,END_THINK+'answer'+EOS])
        events=list(GlmToolReferenceGenerator(model,self.text,max_context=4096).generate_batch_tools(
            [self.messages]*2,TOOLS,max_new_tokens=512))
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop'])
        self.assertEqual([e['row'] for e in events if 'tool_calls' in e],[0]);self.assertTrue(model.states[-1].closed)

    def test_invalid_or_truncated_calls_fail_atomically_and_release(self):
        bad=[self.wire.replace('lookup','unknown'),self.wire.replace('<arg_value>2<','<arg_value>true<'),self.wire[:-2]]
        for batch in (False,True):
            for wire in bad:
                output=END_THINK+wire+EOS
                model=ScriptedGlm(self.text,[output]) if batch else ScriptedModel(self.text,output)
                runner=GlmToolReferenceGenerator(model,self.text,max_context=4096)
                generate=runner.generate_batch_tools if batch else runner.generate_tools
                events=[]
                with self.assertRaises(ValueError):
                    for event in generate([self.messages] if batch else self.messages,TOOLS,max_new_tokens=512):events.append(event)
                self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events))
                self.assertTrue(model.states[-1].closed)

    def test_schema_mutation_cancel_and_primary_error_cleanup(self):
        model=ScriptedModel(self.text,'analysis'+END_THINK+self.wire+EOS)
        runner=GlmToolReferenceGenerator(model,self.text,max_context=4096);tools=copy.deepcopy(TOOLS)
        stream=runner.generate_tools(self.messages,tools,max_new_tokens=512);next(stream)
        tools.clear()
        self.assertEqual(list(stream)[-1]['finish_reason'],'tool_calls')
        stream=runner.generate_tools(self.messages,TOOLS,max_new_tokens=512);next(stream);stream.close()
        self.assertTrue(model.states[-1].closed)
        stream=runner.generate_tools(self.messages,TOOLS,max_new_tokens=512);next(stream);state=model.states[-1]
        with patch.object(model,'forward',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as error:list(stream)
        self.assertTrue(error.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
        runner.cleanup();self.assertTrue(state.closed)

    def test_cli_single_batch_and_invalid_tools_or_history_before_payload(self):
        fixture(self.path,vocab_size=self.text.vocab_size)
        tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        messages=self.path/'messages.json'
        for batch in (False,True):
            messages.write_text(json.dumps([self.messages]*2 if batch else self.messages))
            argv=[str(self.path),'--batch-messages' if batch else '--messages',str(messages),
                  '--tools',str(tools),'--max-new-tokens','512']
            model=ScriptedGlm(self.text,[END_THINK+self.wire+EOS]*2) if batch else ScriptedModel(self.text,END_THINK+self.wire+EOS)
            output=io.StringIO()
            with patch.object(cli,'load_reference_model',return_value=model),redirect_stdout(output):
                self.assertEqual(cli.main(argv),0)
            events=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len([e for e in events if 'tool_calls' in e]),2 if batch else 1)
            for bad in (None,TOOLS*2):
                tools.write_text(json.dumps(bad))
                with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:cli.main(argv)
                    self.assertEqual(error.exception.code,2)
            tools.write_text(json.dumps(TOOLS))
            bad=FIXTURE['cases'][2]['messages'][:-1]
            messages.write_text(json.dumps([bad] if batch else bad))
            with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:cli.main(argv)
                self.assertEqual(error.exception.code,2)
