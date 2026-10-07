import io
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm.glm_flash_tools import FlashTools
from dllm.glm_flash_tool_text import FlashToolText
from dllm.glm_flash_tool_generation import FlashToolReferenceGenerator
from dllm.glm_flash_tool_output import FlashToolTokenStream
from dllm.glm_text import EOS,END_THINK,OBSERVATION
from scripts import run_glm_flash_reference as cli
from tests.test_glm_flash_generation import text_fixture,snapshot
from tests.test_glm_flash_batch_generation import ScriptedBatch,State
from tests.test_glm_flash_tool_generation import TOOLS,CALLS


class FlashBatchToolTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);base=text_fixture(self.path);self.text=FlashToolText(self.path,vocab_size=base.vocab_size)
        self.messages=[dict(role='user',content='Hi')];self.wire=FlashTools(TOOLS).render(CALLS)

    def test_mixed_rows_compact_with_independent_calls_and_finish_reasons(self):
        second=FlashTools(TOOLS).render([dict(type='function',function=dict(name='lookup',arguments='{"value":"123"}'))])
        model=ScriptedBatch(self.text,['r'+END_THINK+self.wire+OBSERVATION,'why'+END_THINK+'answer'+EOS,
                                      'r'+END_THINK+second+EOS,'abcdef'])
        runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        events=list(runner.generate_batch_tools([self.messages]*4,TOOLS,max_new_tokens=[512,512,512,2]))
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop','tool_calls','length'])
        values={e['row']:json.loads(e['tool_calls'][0]['function']['arguments']) for e in events if 'tool_calls' in e}
        self.assertEqual(values,{0:{'value':'false'},2:{'value':'123'}})
        self.assertEqual(''.join(e.get('content','') for e in events if e['row']==1),'answer')
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==3),'ab')
        self.assertTrue(model.states[-1].closed);self.assertTrue(model.states[-1].orders)

    def test_bad_call_or_truncation_fails_whole_batch_without_success(self):
        for output,limit in (('r'+END_THINK+self.wire.replace('lookup','unknown')+EOS,512),
                             ('r'+END_THINK+self.wire+EOS,20)):
            model=ScriptedBatch(self.text,[output,'r'+END_THINK+'ok'+EOS])
            runner=FlashToolReferenceGenerator(model,self.text,max_context=4096);events=[]
            with self.assertRaises(ValueError):
                for e in runner.generate_batch_tools([self.messages]*2,TOOLS,max_new_tokens=limit):events.append(e)
            self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events))
            self.assertTrue(model.states[-1].closed)

    def test_cancel_closes_every_output_buffer_and_owned_state(self):
        streams=[];original=FlashToolTokenStream
        def retained(*args,**kwargs):
            stream=original(*args,**kwargs);streams.append(stream);return stream
        model=ScriptedBatch(self.text,['r'+END_THINK+self.wire+EOS,'x'*512])
        runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        with patch('dllm.glm_flash_tool_generation.FlashToolTokenStream',side_effect=retained):
            generation=runner.generate_batch_tools([self.messages]*2,TOOLS,max_new_tokens=512)
            for _ in range(20):next(generation)
            generation.close()
        self.assertTrue(model.states[-1].closed);self.assertTrue(all(s.parser.closed for s in streams))
        self.assertTrue(all(s.parser.buffer is None or s.parser.buffer.closed for s in streams))

    def test_reorder_failure_and_context_preflight(self):
        model=ScriptedBatch(self.text,['abc','xyz']);runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        with patch.object(State,'reorder',side_effect=RuntimeError('reorder')):
            with self.assertRaisesRegex(RuntimeError,'reorder'):
                list(runner.generate_batch_tools([self.messages]*2,TOOLS,max_new_tokens=[1,3]))
        self.assertTrue(model.states[-1].closed)
        with patch.object(model,'prefill',side_effect=AssertionError('prefill before validation')):
            with self.assertRaises(ValueError):list(runner.generate_batch_tools([self.messages]*2,TOOLS,max_new_tokens=[1]))
            with self.assertRaises(ValueError):list(runner.generate_batch_tools([self.messages,[dict(role='user',content='longer')]],TOOLS,max_new_tokens=3))

    def test_cli_batch_tools_and_actual_loaded_plain_batch(self):
        snapshot(self.path,self.text);tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        messages=self.path/'batch.json';messages.write_text(json.dumps([self.messages]*2))
        argv=[str(self.path),'--batch-messages',str(messages),'--tools',str(tools),'--max-new-tokens','512']
        model=ScriptedBatch(self.text,['r'+END_THINK+self.wire+EOS,'r'+END_THINK+'ok'+EOS]);output=io.StringIO()
        with patch.object(cli,'load_reference_model',return_value=model),redirect_stdout(output):self.assertEqual(cli.main(argv),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop'])
        load=cli.load_reference_model
        def prepared(*args,**kwargs):
            model=load(*args,**kwargs);model.head.zero_();return model
        output=io.StringIO()
        with patch.object(cli,'load_reference_model',side_effect=prepared),redirect_stdout(output):
            self.assertEqual(cli.main(argv[:-1]+['2']),0)
        self.assertEqual([json.loads(line)['row'] for line in output.getvalue().splitlines() if 'finish_reason' in line],[0,1])
        messages.write_text(json.dumps([self.messages,[dict(role='user',content='longer')]]))
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(argv)
            self.assertEqual(error.exception.code,2)
