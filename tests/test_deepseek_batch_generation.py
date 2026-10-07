import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from dllm.deepseek_batch_generation import BatchReferenceGenerator
from dllm.deepseek_generation import ReferenceGenerator
from dllm.deepseek_reference_loader import load_reference_model
from dllm.deepseek_text import EOS, END_THINK
from dllm.floating_model import FloatingModel
from scripts import run_deepseek_reference as cli
from tests.test_deepseek_generation import text_fixture
from tests.test_deepseek_reference_loader import fixture
from tests.test_glm_flash_batch_generation import State, ScriptedBatch


class ScriptedDeepSeek(FloatingModel):
    __init__=ScriptedBatch.__init__
    prefill=ScriptedBatch.prefill
    decode=ScriptedBatch.forward
    def logits(self,state):
        result=torch.zeros(len(state.rows),self.embedding.shape[0])
        for slot,row in enumerate(state.rows):result[slot,self.tokens[row][state.step]]=10
        return result


class DeepSeekBatchTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        self.messages=[dict(role='user',content='Hi')]

    def test_compaction_limits_and_thinking_are_per_row(self):
        for thinking in (False,True):
            prefix='思'+END_THINK if thinking else ''
            model=ScriptedDeepSeek(self.text,[prefix+'a'+EOS,prefix+'uvwxyz',prefix+'long'+EOS])
            runner=BatchReferenceGenerator(model,self.text,max_context=128)
            second_limit=len(self.text.raw_tokens(prefix))+2
            events=list(runner.generate_batch([self.messages]*3,max_new_tokens=[32,second_limit,32],enable_thinking=thinking))
            self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['stop','length','stop'])
            for row,answer in enumerate(('a','uv','long')):
                self.assertEqual(''.join(e.get('content','') for e in events if e['row']==row),answer)
                self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==row),'思' if thinking else '')
            self.assertEqual(model.states[0].rows,[2]);self.assertTrue(model.states[0].closed)

    def test_invalid_batch_and_tools_precede_model(self):
        model=ScriptedDeepSeek(self.text,['abcdef']);runner=BatchReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'prefill',side_effect=AssertionError('prefill')):
            for batch,options in (([],{}),([self.messages]*17,{}),([self.messages],dict(tools=[])),
                    ([self.messages],dict(enable_thinking=1)),([self.messages],dict(max_new_tokens=True)),
                    ([self.messages],dict(max_new_tokens=[0])),
                    ([self.messages,[dict(role='user',content='different')]],{})):
                with self.assertRaises(ValueError):list(runner.generate_batch(batch,**options))

    def test_cancel_decode_cleanup_and_reorder_failures(self):
        model=ScriptedDeepSeek(self.text,['abcdef','uvwxyz']);runner=BatchReferenceGenerator(model,self.text,max_context=128)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);stream.close()
        self.assertTrue(model.states[-1].closed)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);state=model.states[-1]
        with patch.object(model,'decode',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as error:list(stream)
            self.assertTrue(error.exception.__notes__)
            with patch.object(model,'prefill',side_effect=AssertionError('new work')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):
                    list(runner.generate_batch([self.messages]*2,max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed)
        with patch.object(State,'reorder',side_effect=RuntimeError('reorder')):
            with self.assertRaisesRegex(RuntimeError,'reorder'):
                list(runner.generate_batch([self.messages]*2,max_new_tokens=[1,3]))
        self.assertTrue(model.states[-1].closed)

    def test_tiny_packed_logits_match_independent_decode_across_compression_boundary(self):
        fixture(self.path,vocab_size=self.text.vocab_size);model=load_reference_model(self.path)
        rows=[self.messages,[dict(role='user',content='Yo')]]
        prompts=[self.text.chat_tokens(row) for row in rows];length=len(prompts[0])
        token=self.text.raw_tokens('a')[0];references=[]
        for prompt in prompts:
            logits,state=model.prefill(torch.tensor([prompt]),max_tokens=length+5)
            values=[logits.clone()]
            try:
                for step in range(4):values.append(model.decode(torch.tensor([[token]]),state,position=length+step).clone())
            finally:state.close()
            references.append(values)
        self.assertNotEqual(length//4,(length+4)//4)
        original_prefill=model.prefill;original_decode=model.decode;seen=[]
        def controlled(actual,step,rows):
            torch.testing.assert_close(actual,torch.cat([references[row][step] for row in rows]),rtol=5e-5,atol=5e-6)
            seen.append(step);forced=torch.zeros_like(actual);forced[:,token]=10;return forced
        def prefill(tokens,**options):
            logits,state=original_prefill(tokens,**options);return controlled(logits,0,[0,1]),state
        def decode(tokens,state,*,position):
            return controlled(original_decode(tokens,state,position=position),position-length+1,[1])
        runner=BatchReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'prefill',side_effect=prefill),patch.object(model,'decode',side_effect=decode):
            events=list(runner.generate_batch(rows,max_new_tokens=[1,5]))
        self.assertEqual(seen,list(range(5)))
        self.assertEqual([e['usage']['completion_tokens'] for e in events if 'finish_reason' in e],[1,5])
        tiny=BatchReferenceGenerator(model,self.text,max_context=128,max_cache_bytes=1)
        with self.assertRaises(ValueError):list(tiny.generate_batch(rows,max_new_tokens=2))
        self.assertFalse(model.failed_prefills)

    def test_cli_batch_scope_and_invalid_tools_or_ragged_input(self):
        fixture(self.path,vocab_size=self.text.vocab_size);path=self.path/'batch.json'
        path.write_text(json.dumps([self.messages]*2))
        model=ScriptedDeepSeek(self.text,['ab','cd']);output=io.StringIO()
        with patch.object(cli,'load_reference_model',return_value=model),redirect_stdout(output):
            self.assertEqual(cli.main([str(self.path),'--batch-messages',str(path),'--max-new-tokens','2','--context','128']),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(events[0]['official_model_qualified'])
        self.assertEqual([e['row'] for e in events if 'finish_reason' in e],[0,1])
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main([str(self.path),'--batch-messages',str(path),'--tools','missing.json'])
            self.assertEqual(error.exception.code,2)
            path.write_text(json.dumps([self.messages,[dict(role='user',content='longer')]]))
            with self.assertRaises(SystemExit) as error:
                cli.main([str(self.path),'--batch-messages',str(path)])
            self.assertEqual(error.exception.code,2)

    def test_cli_preserves_primary_failure_if_cleanup_also_fails(self):
        fixture(self.path,vocab_size=self.text.vocab_size)
        model=ScriptedDeepSeek(self.text,['abc']);runner=ReferenceGenerator(model,self.text,max_context=128)
        with patch.object(cli,'load_reference_model',return_value=model),patch.object(cli,'ReferenceGenerator',return_value=runner), \
                patch.object(runner,'generate',side_effect=RuntimeError('primary decode')), \
                patch.object(runner,'cleanup',side_effect=RuntimeError('cleanup')), \
                redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):cli.main([str(self.path),'--prompt','Hi','--max-new-tokens','2','--context','128'])
        self.assertIn('primary decode',errors.getvalue())
