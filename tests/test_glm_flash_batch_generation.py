import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch

from dllm.glm_flash_batch_generation import FlashBatchReferenceGenerator
from dllm.glm_flash_generation import FlashReferenceGenerator
from dllm.glm_flash_model import FlashModel
from dllm.glm_flash_reference_loader import load_reference_model
from dllm.glm_text import END_THINK, EOS
from scripts import run_glm_flash_reference as cli
from tests.test_glm_flash_generation import text_fixture, snapshot


class State:
    def __init__(self, batch, position):
        self.rows=list(range(batch));self.position=position;self.step=0
        self.closed=False;self.orders=[]
    def reorder(self, indices):
        self.orders.append(indices.tolist());self.rows=[self.rows[i] for i in indices.tolist()]
    def close(self):self.closed=True


class ScriptedBatch(FlashModel):
    def __init__(self, text, completions):
        self.embedding=torch.zeros(text.vocab_size,1)
        self.tokens=[text.raw_tokens(value) for value in completions]
        self.states=[];self.failed_cleanups=[];self.lock=threading.RLock();self.inputs=[]
    def logits(self,state):
        result=torch.zeros(len(state.rows),self.embedding.shape[0])
        for slot,row in enumerate(state.rows):result[slot,self.tokens[row][state.step]]=10
        result[:,-1]=100  # Unmapped padded vocabulary rows must not win.
        return result
    def prefill(self,tokens,**options):
        state=State(len(tokens),tokens.shape[1]);self.states.append(state)
        return self.logits(state),state
    def forward(self,tokens,state,*,position):
        assert position==state.position
        self.inputs.append((list(state.rows),tokens.flatten().tolist()))
        state.position+=1;state.step+=1
        return self.logits(state)


class FlashBatchTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        self.messages=[dict(role='user',content='Hi')]

    def test_independent_channels_limits_and_compaction_preserve_original_rows(self):
        model=ScriptedBatch(self.text,[END_THINK+'a'+EOS, 'uvwxyz', '思'+END_THINK+'answer'+EOS])
        runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate_batch([self.messages]*3,max_new_tokens=[12,2,30]))
        finals=[event for event in events if 'finish_reason' in event]
        self.assertEqual([event['row'] for event in finals],[0,1,2])
        self.assertEqual([event['finish_reason'] for event in finals],['stop','length','stop'])
        self.assertEqual([event['usage']['completion_tokens'] for event in finals],[3,2,len(model.tokens[2])])
        self.assertEqual(''.join(e.get('content','') for e in events if e['row']==0),'a')
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==1),'uv')
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events if e['row']==2),'思')
        self.assertEqual(''.join(e.get('content','') for e in events if e['row']==2),'answer')
        self.assertEqual(model.states[0].orders,[[0,2],[1]])
        self.assertEqual(model.inputs[-1][0],[2]);self.assertTrue(model.states[0].closed)

    def test_invalid_whole_batch_precedes_prefill(self):
        model=ScriptedBatch(self.text,['abc']);runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        cases=[([],2),([self.messages]*17,2),([self.messages],True),([self.messages],[0]),
               ([self.messages],[1,2]),([self.messages],128),
               ([self.messages,[dict(role='user',content='longer')]],2)]
        with patch.object(model,'prefill',side_effect=AssertionError('prefill')):
            for batch,limit in cases:
                with self.subTest(limit=limit),self.assertRaises(ValueError):
                    list(runner.generate_batch(batch,max_new_tokens=limit))

    def test_cancellation_primary_failure_and_retained_cleanup(self):
        model=ScriptedBatch(self.text,['abcdef','uvwxyz'])
        runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);stream.close()
        self.assertTrue(model.states[-1].closed)
        stream=runner.generate_batch([self.messages]*2,max_new_tokens=4);next(stream);state=model.states[-1]
        with patch.object(model,'forward',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as error:list(stream)
            self.assertTrue(error.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(model,'prefill',side_effect=AssertionError('new work')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):
                    list(runner.generate_batch([self.messages]*2,max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed)

    def test_reorder_failure_and_nonfinite_logits_release_whole_batch(self):
        model=ScriptedBatch(self.text,['abcd','wxyz']);runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        with patch.object(State,'reorder',side_effect=RuntimeError('reorder')):
            with self.assertRaisesRegex(RuntimeError,'reorder'):
                list(runner.generate_batch([self.messages]*2,max_new_tokens=[1,3]))
        self.assertTrue(model.states[-1].closed)
        with patch.object(model,'logits',return_value=torch.full((2,self.text.vocab_size),float('nan'))):
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                list(runner.generate_batch([self.messages]*2,max_new_tokens=2))
        self.assertTrue(model.states[-1].closed)

    def test_final_success_requires_cleanup(self):
        model=ScriptedBatch(self.text,['abcdef','uvwxyz']);runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        events=[]
        with patch.object(State,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'cleanup'):
                for event in runner.generate_batch([self.messages]*2,max_new_tokens=2):events.append(event)
        self.assertFalse(any('finish_reason' in event for event in events))
        runner.cleanup();self.assertTrue(model.states[-1].closed)

    def test_loaded_tiny_batch_matches_independent_rows_and_budget_fails_before_compute(self):
        snapshot(self.path,self.text);model=load_reference_model(self.path);model.head.zero_()
        runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        rows=[self.messages,[dict(role='user',content='Yo')]]
        events=list(runner.generate_batch(rows,max_new_tokens=[1,3]))
        for row,limit in enumerate([1,3]):
            single=list(FlashReferenceGenerator(model,self.text,max_context=128).generate(rows[row],max_new_tokens=limit))
            actual=[{k:v for k,v in event.items() if k!='row'} for event in events if event['row']==row]
            self.assertEqual(actual,single)
        tiny=FlashBatchReferenceGenerator(model,self.text,max_context=128,max_cache_bytes=1)
        with patch.object(model,'forward',side_effect=AssertionError('compute before admission')):
            with self.assertRaises(ValueError):list(tiny.generate_batch(rows,max_new_tokens=2))

    def test_cli_batch_and_ragged_preflight(self):
        snapshot(self.path,self.text);path=self.path/'batch.json'
        path.write_text(json.dumps([self.messages]*2))
        load=cli.load_reference_model
        def prepared(*args,**kwargs):
            model=load(*args,**kwargs);model.head.zero_();return model
        output=io.StringIO()
        with patch.object(cli,'load_reference_model',side_effect=prepared),redirect_stdout(output):
            self.assertEqual(cli.main([str(self.path),'--batch-messages',str(path),'--max-new-tokens','2','--context','128']),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(events[0]['official_model_qualified'])
        self.assertEqual([e['row'] for e in events if 'finish_reason' in e],[0,1])
        path.write_text(json.dumps([self.messages,[dict(role='user',content='longer')]]))
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main([str(self.path),'--batch-messages',str(path),'--max-new-tokens','2'])
            self.assertEqual(error.exception.code,2)

    def test_real_batch_logits_match_independent_forced_decode_after_compaction(self):
        snapshot(self.path,self.text);model=load_reference_model(self.path)
        rows=[self.messages,[dict(role='user',content='Yo')]]
        prompts=[self.text.chat_tokens(row) for row in rows]
        length=len(prompts[0]);token=self.text.raw_tokens('a')[0];references=[]
        for prompt in prompts:
            logits,state=model.prefill(torch.tensor([prompt]),capacity=length+3)
            values=[logits.clone()]
            try:
                for step in range(2):
                    values.append(model.forward(torch.tensor([[token]]),state,position=length+step).clone())
            finally:state.close()
            references.append(values)
        forward=model.forward;seen=[]
        def compare(tokens,state,*,position,all_logits=False):
            actual=forward(tokens,state,position=position,all_logits=all_logits)
            step=0 if position==0 else position-length+1
            expected=torch.cat([references[row][step] for row in ([0,1] if step==0 else [1])])
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
            seen.append(step)
            forced=torch.zeros_like(actual);forced[:,token]=10
            return forced
        runner=FlashBatchReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'forward',side_effect=compare):
            events=list(runner.generate_batch(rows,max_new_tokens=[1,3]))
        self.assertEqual(seen,[0,1,2])
        self.assertEqual(len([e for e in events if 'finish_reason' in e]),2)
