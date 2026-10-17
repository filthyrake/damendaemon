import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from damendaemon.deepseek_batch_generation import BatchReferenceGenerator
from damendaemon.deepseek_text import EOS, END_THINK
from damendaemon.deepseek_tools import render_tool_calls
from scripts import run_deepseek_reference as cli
from tests.test_deepseek_batch_generation import ScriptedDeepSeek
from tests.test_deepseek_generation import text_fixture,ScriptedModel
from damendaemon.deepseek_generation import ReferenceGenerator
from tests.test_deepseek_reference_loader import fixture


def calls(name='lookup', value='first'):
    return [dict(type='function',function=dict(name=name,arguments=json.dumps({'value':value},ensure_ascii=False)))]


def completion(name='lookup', value='first'):
    return '\n\n'+render_tool_calls(calls(name,value))+EOS


class DeepSeekBatchToolTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        self.messages=[dict(role='user',content='Hi')]
        self.tools=[dict(type='function',function=dict(name='lookup'))]

    def test_mixed_tool_rows_keep_arguments_identity_and_finish_reason(self):
        for thinking in (False,True):
            prefix='reason'+END_THINK if thinking else ''
            model=ScriptedDeepSeek(self.text,[prefix+completion(value='甲'),prefix+'answer'+EOS,prefix+completion(value='second')])
            runner=BatchReferenceGenerator(model,self.text,max_context=4096)
            events=list(runner.generate_batch([self.messages]*3,max_new_tokens=512,tools=self.tools,enable_thinking=thinking))
            self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop','tool_calls'])
            emitted={e['row']:e['tool_calls'] for e in events if 'tool_calls' in e}
            self.assertEqual(set(emitted),{0,2})
            for row,value in ((0,'甲'),(2,'second')):
                self.assertEqual(json.loads(emitted[row][0]['function']['arguments']),{'value':value})
                self.assertEqual(emitted[row][0]['function']['name'],'lookup')
            self.assertEqual(''.join(e.get('content','') for e in events if e['row']==1),'answer')
            for row in range(3):
                self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==row),'reason' if thinking else '')
            self.assertTrue(model.states[-1].closed);self.assertIn([0,2],model.states[-1].orders)

    def test_undeclared_or_truncated_calls_emit_no_calls_or_success(self):
        for output,limit in ((completion('unknown'),512),(completion(),20),(completion().replace('string="true"','string="oops"'),512)):
            model=ScriptedDeepSeek(self.text,[output,'normal'+EOS]);runner=BatchReferenceGenerator(model,self.text,max_context=4096)
            events=[]
            with self.assertRaises(ValueError):
                for event in runner.generate_batch([self.messages]*2,max_new_tokens=limit,tools=self.tools):events.append(event)
            self.assertFalse(any('tool_calls' in event or 'finish_reason' in event for event in events))
            self.assertTrue(model.states[-1].closed)

    def test_declaration_snapshot_cannot_be_changed_after_stream_starts(self):
        tools=list(self.tools);model=ScriptedDeepSeek(self.text,['lead'+completion('unknown')])
        runner=BatchReferenceGenerator(model,self.text,max_context=4096)
        stream=runner.generate_batch([self.messages],max_new_tokens=512,tools=tools)
        next(stream)
        tools.append(dict(type='function',function=dict(name='unknown')))
        with self.assertRaisesRegex(ValueError,'undeclared'):list(stream)
        self.assertTrue(model.states[-1].closed)

    def test_single_request_cannot_add_declaration_after_stream_starts(self):
        tools=[dict(type='function',function=dict(name='lookup'))]
        model=ScriptedModel(self.text,'lead'+completion('unknown'))
        runner=ReferenceGenerator(model,self.text,max_context=4096)
        stream=runner.generate(self.messages,max_new_tokens=512,tools=tools);next(stream)
        tools.append(dict(type='function',function=dict(name='unknown')))
        emitted=[]
        with self.assertRaisesRegex(ValueError,'undeclared'):
            for event in stream:emitted.append(event)
        self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in emitted))
        self.assertTrue(model.states[-1].closed)
        # The updated declaration is allowed on a subsequent, freshly rendered request.
        events=list(runner.generate(self.messages,max_new_tokens=512,tools=tools))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls');self.assertTrue(model.states[-1].closed)

    def test_single_request_keeps_original_declarations_after_removal_or_rename(self):
        for mutate in (lambda tools:tools.clear(),lambda tools:tools[0]['function'].update(name='other')):
            tools=[dict(type='function',function=dict(name='lookup'))]
            model=ScriptedModel(self.text,'lead'+completion('lookup'))
            runner=ReferenceGenerator(model,self.text,max_context=4096)
            stream=runner.generate(self.messages,max_new_tokens=512,tools=tools);next(stream);mutate(tools)
            events=list(stream);self.assertEqual(events[-1]['finish_reason'],'tool_calls')
            self.assertEqual([c['function']['name'] for e in events for c in e.get('tool_calls',[])],['lookup'])
            self.assertTrue(model.states[-1].closed)

    def test_cancel_while_another_row_buffers_tool_block_releases_state(self):
        model=ScriptedDeepSeek(self.text,[completion(),'x'*512]);runner=BatchReferenceGenerator(model,self.text,max_context=4096)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=512,tools=self.tools)
        for _ in range(20):next(stream)
        stream.close();self.assertTrue(model.states[-1].closed)
        events=list(runner.generate_batch([self.messages]*2,max_new_tokens=512,tools=self.tools))
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','length'])

    def test_cli_tool_batch_and_invalid_declarations_before_payload(self):
        fixture(self.path,vocab_size=self.text.vocab_size)
        messages=self.path/'batch.json';messages.write_text(json.dumps([self.messages]*2))
        tools=self.path/'tools.json';tools.write_text(json.dumps(self.tools))
        model=ScriptedDeepSeek(self.text,[completion(),'answer'+EOS]);output=io.StringIO()
        argv=[str(self.path),'--batch-messages',str(messages),'--tools',str(tools),'--max-new-tokens','512']
        with patch.object(cli,'load_reference_model',return_value=model),redirect_stdout(output):
            self.assertEqual(cli.main(argv),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop'])
        self.assertEqual([e['row'] for e in events if 'tool_calls' in e],[0])
        tools.write_text(json.dumps(self.tools*2))
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(argv)
            self.assertEqual(error.exception.code,2)
