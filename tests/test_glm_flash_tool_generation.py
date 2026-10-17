import io
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from damendaemon.glm_flash_tools import FlashTools
from damendaemon.glm_flash_tool_output import FlashToolOutput,FlashToolTokenStream
from damendaemon.glm_flash_tool_text import FlashToolText
from damendaemon.glm_flash_tool_generation import FlashToolReferenceGenerator
from damendaemon.glm_text import EOS,OBSERVATION,USER,END_THINK,MAX_TEXT_BYTES
from scripts import run_glm_flash_reference as cli
from tests.test_glm_flash_generation import text_fixture,ScriptedFlash,snapshot

TOOLS=[dict(type='function',function=dict(name='lookup',parameters=dict(type='object',properties={'value':{'type':'string'}},required=['value'],additionalProperties=False)))]
CALLS=[dict(type='function',function=dict(name='lookup',arguments='{"value":"false"}'))]


class FlashToolGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);base=text_fixture(self.path)
        self.text=FlashToolText(self.path,vocab_size=base.vocab_size)
        self.messages=[dict(role='user',content='Hi')];self.codec=FlashTools(TOOLS)
        self.wire=self.codec.render(CALLS)

    def test_every_split_keeps_channels_and_atomic_typed_calls(self):
        for stop in (EOS,OBSERVATION,USER):
            output='分析🦙'+END_THINK+'checking'+self.wire+stop
            for split in range(len(output)+1):
                parser=FlashToolOutput(self.codec)
                events=parser.feed(output[:split])+parser.feed(output[split:])
                self.assertFalse(any('tool_calls' in e for e in events))
                events+=parser.finish()
                self.assertEqual(''.join(e.get('reasoning_content','') for e in events),'分析🦙')
                self.assertEqual(''.join(e.get('content','') for e in events),'checking')
                calls=[e['tool_calls'] for e in events if 'tool_calls' in e]
                self.assertEqual(len(calls),1)
                self.assertEqual(json.loads(calls[0][0]['function']['arguments']),{'value':'false'})
                self.assertTrue(parser.buffer.closed)

    def test_token_stream_handles_utf8_and_plain_output(self):
        for output,expected in (('分析🦙'+END_THINK+'answer'+EOS,'answer'),('analysis'+END_THINK+self.wire+EOS,'')):
            stream=FlashToolTokenStream(self.text,self.codec);events=[]
            for token in self.text.raw_tokens(output):events+=stream.feed(token)
            events+=stream.finish()
            self.assertEqual(''.join(e.get('content','') for e in events),expected)
            self.assertTrue(stream.parser.closed)

    def test_malformed_truncated_wrong_channel_and_bounds_fail_closed(self):
        for output in (self.wire+EOS,END_THINK+self.wire,END_THINK+self.wire+EOS+'trailing',
                       END_THINK+self.wire.replace('lookup','unknown')+EOS,
                       END_THINK+self.wire.replace('<arg_value>false','<arg_value><|image|>')+EOS,
                       END_THINK+'<tool_ca'):
            parser=FlashToolOutput(self.codec)
            with self.assertRaises(ValueError):
                parser.feed(output);parser.finish(truncated=True)
            self.assertTrue(parser.failed)
        parser=FlashToolOutput(self.codec)
        with self.assertRaises(ValueError):parser.feed('x'*(MAX_TEXT_BYTES+1))
        with self.assertRaises(RuntimeError):parser.feed('again')

    def test_owned_generation_tool_finish_usage_and_abandonment(self):
        completion='analysis'+END_THINK+self.wire+OBSERVATION
        model=ScriptedFlash(self.text,completion);runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        events=list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=512))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        self.assertEqual(events[-1]['usage']['completion_tokens'],len(model.tokens))
        self.assertTrue(model.states[-1].closed)
        self.assertEqual([e['tool_calls'][0]['function']['name'] for e in events if 'tool_calls' in e],['lookup'])
        stream=runner.generate_tools(self.messages,TOOLS,max_new_tokens=512);next(stream);stream.close()
        self.assertTrue(model.states[-1].closed)
        with patch.object(model,'forward',side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError,'decode failed'):list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=512))
        self.assertTrue(model.states[-1].closed)

    def test_parser_and_cleanup_failure_retains_primary_and_blocks_new_prefill(self):
        model=ScriptedFlash(self.text,'analysis'+END_THINK+self.wire.replace('lookup','unknown')+EOS)
        runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        stream=runner.generate_tools(self.messages,TOOLS,max_new_tokens=512);next(stream);state=model.states[-1]
        with patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(ValueError,'undeclared') as error:list(stream)
            self.assertTrue(error.exception.__notes__)
            with patch.object(model,'prefill',side_effect=AssertionError('new work')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=512))
        runner.cleanup();self.assertTrue(state.closed)

    def test_cli_tool_front_door_preflight_and_tiny_loaded_plain_reply(self):
        snapshot(self.path,self.text);tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        model=ScriptedFlash(self.text,'r'+END_THINK+self.wire+EOS);out=io.StringIO()
        argv=[str(self.path),'--prompt','Hi','--tools',str(tools),'--max-new-tokens','512']
        with patch.object(cli,'load_reference_model',return_value=model),redirect_stdout(out):
            self.assertEqual(cli.main(argv),0)
        events=[json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(events[-1]['finish_reason'],'tool_calls');self.assertFalse(events[0]['official_model_qualified'])
        load=cli.load_reference_model
        def prepared(*args,**kwargs):
            result=load(*args,**kwargs);result.head.zero_();return result
        out=io.StringIO()
        with patch.object(cli,'load_reference_model',side_effect=prepared),redirect_stdout(out):
            self.assertEqual(cli.main([str(self.path),'--prompt','Hi','--tools',str(tools),'--max-new-tokens','2']),0)
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])['finish_reason'],'length')
        tools.write_text('[]')
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(argv)
            self.assertEqual(error.exception.code,2)
            with self.assertRaises(SystemExit) as error:
                cli.main([str(self.path),'--batch-messages','missing.json','--tools',str(tools)])
            self.assertEqual(error.exception.code,2)

    def test_returned_calls_can_resume_with_identity_matched_results(self):
        model=ScriptedFlash(self.text,'reason'+END_THINK+self.wire+OBSERVATION)
        runner=FlashToolReferenceGenerator(model,self.text,max_context=4096)
        events=list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=512))
        calls=next(event['tool_calls'] for event in events if 'tool_calls' in event)
        history=self.messages+[dict(role='assistant',content='',reasoning_content='reason',tool_calls=calls),
                               dict(role='tool',tool_call_id=calls[0]['id'],content='result data')]
        following=ScriptedFlash(self.text,'inspect'+END_THINK+'done'+EOS)
        runner=FlashToolReferenceGenerator(following,self.text,max_context=4096)
        events=list(runner.generate_tools(history,TOOLS,max_new_tokens=128,clear_thinking=True))
        self.assertEqual(''.join(e.get('content','') for e in events),'done')
        self.assertEqual(events[-1]['finish_reason'],'stop');self.assertTrue(following.states[-1].closed)
        bounded=FlashToolReferenceGenerator(following,self.text,max_context=128)
        with patch.object(following,'prefill',side_effect=AssertionError('prefill before context admission')):
            with self.assertRaises(ValueError):list(bounded.generate_tools(history,TOOLS,max_new_tokens=128))
