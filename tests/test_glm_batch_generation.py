import io
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from damendaemon.glm_model import GlmModel
from damendaemon.glm_generation import ReferenceGenerator
from damendaemon.glm_batch_generation import BatchReferenceGenerator
from damendaemon.glm_reference_loader import load_reference_model
from damendaemon.glm_text import END_THINK,EOS
from scripts import run_glm_reference as cli
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture,floating_control
from tests.test_glm_flash_batch_generation import ScriptedBatch,State


class ScriptedGlm(GlmModel):
    __init__=ScriptedBatch.__init__
    logits=ScriptedBatch.logits
    prefill=ScriptedBatch.prefill
    forward=ScriptedBatch.forward


class GlmBatchTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        self.messages=[dict(role='user',content='Hi')]

    def test_independent_channels_limits_and_compaction(self):
        model=ScriptedGlm(self.text,[END_THINK+'a'+EOS,'uvwxyz','思'+END_THINK+'answer'+EOS])
        events=list(BatchReferenceGenerator(model,self.text,max_context=128).generate_batch(
            [self.messages]*3,max_new_tokens=[12,2,30]))
        finals=[e for e in events if 'finish_reason' in e]
        self.assertEqual([e['row'] for e in finals],[0,1,2])
        self.assertEqual([e['finish_reason'] for e in finals],['stop','length','stop'])
        self.assertEqual([e['usage']['completion_tokens'] for e in finals],[3,2,len(model.tokens[2])])
        self.assertEqual(''.join(e.get('content','') for e in events if e['row']==0),'a')
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==1),'uv')
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==2),'思')
        self.assertEqual(''.join(e.get('content','') for e in events if e['row']==2),'answer')
        self.assertEqual(model.states[0].orders,[[0,2],[1]])
        self.assertTrue(model.states[0].closed)

    def test_preflight_before_prefill_and_cli_payload(self):
        model=ScriptedGlm(self.text,['abc']);runner=BatchReferenceGenerator(model,self.text,max_context=128)
        cases=[([],2),([self.messages]*17,2),([self.messages],True),([self.messages],[0]),
               ([self.messages],[1,2]),([self.messages],128),([self.messages,[dict(role='user',content='longer')]],2)]
        with patch.object(model,'prefill',side_effect=AssertionError('prefill')):
            for rows,limit in cases:
                with self.assertRaises(ValueError):list(runner.generate_batch(rows,max_new_tokens=limit))
        fixture(self.path,vocab_size=self.text.vocab_size)
        path=self.path/'batch.json';path.write_text(json.dumps([self.messages,[dict(role='user',content='longer')]]))
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main([str(self.path),'--batch-messages',str(path)])
            self.assertEqual(error.exception.code,2)

    def test_cancellation_failure_and_cleanup_retry(self):
        model=ScriptedGlm(self.text,['abcdef','uvwxyz']);runner=BatchReferenceGenerator(model,self.text,max_context=128)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);stream.close()
        self.assertTrue(model.states[-1].closed)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);state=model.states[-1]
        with patch.object(model,'forward',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as error:list(stream)
            self.assertTrue(error.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(model,'prefill',side_effect=AssertionError('new work')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):list(runner.generate_batch([self.messages],max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed)

    def test_failed_reorder_nonfinite_or_final_cleanup_emits_no_success(self):
        for target,error in (('reorder','reorder'),('logits','nonfinite'),('close','cleanup')):
            model=ScriptedGlm(self.text,['abcd','wxyz']);runner=BatchReferenceGenerator(model,self.text,max_context=128)
            patcher=(patch.object(model,'logits',return_value=torch.full((2,self.text.vocab_size),float('nan'))) if target=='logits'
                     else patch.object(State,target,side_effect=RuntimeError(error)))
            events=[]
            with patcher:
                with self.assertRaisesRegex(RuntimeError,error):
                    for event in runner.generate_batch([self.messages]*2,max_new_tokens=[1,3]):events.append(event)
            self.assertFalse(any('finish_reason' in e for e in events))
            runner.cleanup();self.assertTrue(model.states[-1].closed)

    def test_packed_batch_logits_match_independent_floating_rows_after_compaction(self):
        config,_,values=fixture(self.path,vocab_size=self.text.vocab_size)
        model=load_reference_model(self.path);control=floating_control(config,values)
        rows=[self.messages,[dict(role='user',content='Yo')]]
        prompts=[self.text.chat_tokens(row) for row in rows];length=len(prompts[0])
        token=self.text.raw_tokens('a')[0];references=[]
        for prompt in prompts:
            logits,state=control.prefill(torch.tensor([prompt]),capacity=length+3)
            expected=[logits.clone()]
            try:
                for step in range(2):expected.append(control.forward(torch.tensor([[token]]),state,position=length+step).clone())
            finally:state.close()
            references.append(expected)
        forward=model.forward;seen=[]
        def compare(tokens,state,*,position,all_logits=False):
            actual=forward(tokens,state,position=position,all_logits=all_logits)
            step=0 if position==0 else position-length+1
            expected=torch.cat([references[row][step] for row in ([0,1] if step==0 else [1])])
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
            seen.append(step);forced=torch.zeros_like(actual);forced[:,token]=10;return forced
        with patch.object(model,'forward',side_effect=compare):
            events=list(BatchReferenceGenerator(model,self.text,max_context=128).generate_batch(rows,max_new_tokens=[1,3]))
        self.assertEqual(seen,[0,1,2]);self.assertEqual(len([e for e in events if 'finish_reason' in e]),2)
        tiny=BatchReferenceGenerator(model,self.text,max_context=128,max_cache_bytes=1)
        with patch.object(model,'forward',side_effect=AssertionError('compute')):
            with self.assertRaises(ValueError):list(tiny.generate_batch(rows,max_new_tokens=2))

    def test_loaded_cli_batch_matches_single_streams(self):
        fixture(self.path,vocab_size=self.text.vocab_size);path=self.path/'batch.json'
        path.write_text(json.dumps([self.messages]*2));load=cli.load_reference_model
        def prepared(*args,**kwargs):
            model=load(*args,**kwargs);model.head.zero_();return model
        output=io.StringIO()
        with patch.object(cli,'load_reference_model',side_effect=prepared),redirect_stdout(output):
            self.assertEqual(cli.main([str(self.path),'--batch-messages',str(path),'--max-new-tokens','2','--context','128']),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        model=prepared(self.path);expected=list(ReferenceGenerator(model,self.text,max_context=128).generate(self.messages,max_new_tokens=2))
        for row in (0,1):
            self.assertEqual([{k:v for k,v in e.items() if k!='row'} for e in events[1:] if e['row']==row],expected)
        self.assertFalse(events[0]['official_model_qualified'])
