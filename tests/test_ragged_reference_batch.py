from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from damendaemon.reference_batch import prepare_batch
from tests import test_glm_batch_generation as glm
from tests import test_glm_flash_batch_generation as flash
from tests import test_deepseek_batch_generation as deepseek


class RaggedReferenceTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def families(self):
        for family in (glm,flash,deepseek):
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)
            text=family.text_fixture(path)
            if family is flash:family.snapshot(path,text)
            else:family.fixture(path,vocab_size=text.vocab_size)
            single=family.FlashReferenceGenerator if family is flash else family.ReferenceGenerator
            batch=family.FlashBatchReferenceGenerator if family is flash else family.BatchReferenceGenerator
            yield family,path,text,single,batch

    def test_three_loaded_families_match_independent_row_logits_and_events(self):
        rows=[[dict(role='user',content=s)] for s in ('Hi','longer input','Yo')];limits=[1,2,3]
        for family,path,text,single,batch in self.families():
            with self.subTest(family=family.__name__):
                model=family.load_reference_model(path);token=text.raw_tokens('a')[0]
                prompts=[text.chat_tokens(row) for row in rows]
                self.assertEqual(len(prompts[0]),len(prompts[2]));self.assertNotEqual(len(prompts[0]),len(prompts[1]))
                prefill=model.prefill;decode=model.decode if family is deepseek else model.forward
                expected=[];logits_by_row=[]
                for row,limit in enumerate(limits):
                    options=dict(max_tokens=len(prompts[row])+limit) if family is deepseek else dict(capacity=len(prompts[row])+limit)
                    logits,state=prefill(torch.tensor([prompts[row]]),**options);values=[logits.clone()]
                    try:
                        for step in range(limit-1):
                            values.append(decode(torch.tensor([[token]]),state,position=len(prompts[row])+step).clone())
                    finally:state.close()
                    logits_by_row.append(values)
                def forced(logits):
                    result=torch.zeros_like(logits);result[:,token]=10;return result
                decode_name='decode' if family is deepseek else 'forward'
                def original_prefill(*args,**kwargs):
                    with patch.object(model,decode_name,side_effect=decode):return prefill(*args,**kwargs)
                with patch.object(model,'prefill',side_effect=lambda *a,**k:(lambda v:(forced(v[0]),v[1]))(original_prefill(*a,**k))),\
                     patch.object(model,decode_name,side_effect=lambda *a,**k:forced(decode(*a,**k))):
                    for row,limit in enumerate(limits):expected.append(list(single(model,text,max_context=128).generate(rows[row],max_new_tokens=limit)))
                cohorts=iter(([0,2],[1]));active=[];step=0;previous=[];calls=[]
                def checked_prefill(tokens,**options):
                    nonlocal active,step
                    # Previous cache must be closed before allocating another cohort.
                    if previous:self.assertTrue(previous[-1].closed)
                    active=list(next(cohorts));step=0;calls.append(active[:])
                    self.assertEqual(tokens.tolist(),[prompts[row] for row in active])
                    logits,state=original_prefill(tokens,**options);previous.append(state)
                    torch.testing.assert_close(logits,torch.cat([logits_by_row[row][0] for row in active]),rtol=5e-5,atol=5e-6)
                    return forced(logits),state
                def checked_decode(tokens,state,*,position):
                    nonlocal active,step
                    step+=1;active=[row for row in active if limits[row]>step]
                    logits=decode(tokens,state,position=position)
                    torch.testing.assert_close(logits,torch.cat([logits_by_row[row][step] for row in active]),rtol=5e-5,atol=5e-6)
                    return forced(logits)
                with patch.object(model,'prefill',side_effect=checked_prefill),patch.object(model,decode_name,side_effect=checked_decode):
                    events=list(batch(model,text,max_context=128).generate_batch(rows,max_new_tokens=limits,ragged=True))
                self.assertEqual(calls,[[0,2],[1]])
                self.assertTrue(all(state.closed for state in previous))
                self.assertEqual([e['row'] for e in events if 'finish_reason' in e],[0,1,2])
                for row in range(3):
                    self.assertEqual([{k:v for k,v in e.items() if k!='row'} for e in events if e['row']==row],expected[row])

    def test_nested_bytelevel_decoders_work_through_generation_and_host_admission(self):
        from tokenizers import decoders
        from damendaemon.native_host_generation import snapshot_text
        from damendaemon.deepseek_host_generation import _snapshot
        rows = [[dict(role='user', content='Hi')]]
        for family, path, text, single, batch in self.families():
            model = family.load_reference_model(path)
            def output():
                return (list(single(model, text, max_context=128).generate(
                            rows[0], max_new_tokens=2)),
                        list(batch(model, text, max_context=128).generate_batch(
                            rows, max_new_tokens=2)))
            expected = output()
            text.tokenizer.decoder = decoders.Sequence([
                decoders.Sequence([decoders.ByteLevel()])])
            with self.subTest(family=family.__name__):
                self.assertEqual(output(), expected)
                copied = _snapshot(text) if family is deepseek else snapshot_text(text, type(text))
                self.assertEqual(copied.tokenizer.decode([0]), text.tokenizer.decode([0]))
                text.tokenizer.decoder = decoders.Sequence([
                    decoders.ByteLevel(), decoders.Replace('a', 'b')])
                with self.assertRaises(ValueError):
                    single(model, text, max_context=128)
                with self.assertRaises(ValueError):
                    if family is deepseek:
                        _snapshot(text)
                    else:
                        snapshot_text(text, type(text))

    def test_rejected_generation_logits_are_execution_errors_and_release_cache(self):
        rows=[[dict(role='user',content='Hi')]]
        for family,path,text,single,batch in self.families():
            model=family.load_reference_model(path)
            original=model.prefill
            for batched in (False,True):
                for malformed in ('nan','shape','dtype'):
                    states=[]
                    def prefill(*args,**kwargs):
                        logits,state=original(*args,**kwargs)
                        states.append(state)
                        if malformed=='nan':logits=torch.full_like(logits,float('nan'))
                        elif malformed=='shape':logits=logits[:,:-1]
                        else:logits=logits.double()
                        return logits,state
                    runner=(batch if batched else single)(model,text,max_context=128)
                    with self.subTest(family=family.__name__,batched=batched,malformed=malformed), \
                            patch.object(model,'prefill',side_effect=prefill):
                        stream=(runner.generate_batch(rows,max_new_tokens=1) if batched
                                else runner.generate(rows[0],max_new_tokens=1))
                        with self.assertRaisesRegex(RuntimeError,'generation logits'):
                            next(stream)
                    self.assertEqual(len(states),1)
                    self.assertTrue(states[0].closed)

    def test_commands_opt_in_and_preflight_all_rows(self):
        rows=[[dict(role='user',content=s)] for s in ('Hi','longer input')]
        for family,path,text,_,_ in self.families():
            with self.subTest(family=family.__name__):
                source=path/'batch.json';source.write_text(json.dumps(rows));load=family.cli.load_reference_model
                def prepared(*args,**kwargs):
                    model=load(*args,**kwargs);(model.head['head.weight'] if isinstance(model.head,dict) else model.head).zero_();return model
                output=io.StringIO()
                with patch.object(family.cli,'load_reference_model',side_effect=prepared),redirect_stdout(output):
                    self.assertEqual(family.cli.main([str(path),'--batch-messages',str(source),'--ragged-batch',
                        '--max-new-tokens','2','--context','128']),0)
                events=[json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([e['row'] for e in events if 'finish_reason' in e],[0,1])
                self.assertEqual([e['usage']['prompt_tokens'] for e in events if 'finish_reason' in e],[len(text.chat_tokens(row)) for row in rows])
                invalid=[['--prompt','Hi','--ragged-batch'],['--batch-messages',str(source),'--ragged-batch','--tools','missing'],
                         ['--batch-messages',str(source),'--ragged-batch','--context','4']]
                for options in invalid:
                    with patch.object(family.cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:family.cli.main([str(path),*options])
                        self.assertEqual(error.exception.code,2)

    def test_per_row_context_limit_and_explicit_boolean(self):
        from types import SimpleNamespace
        text=SimpleNamespace(chat_tokens=lambda row:row)
        prompts,limits=prepare_batch(text,[[1],[1]*8],[8,1],9,ragged=True)
        self.assertEqual(limits,[8,1]);self.assertEqual(list(map(len,prompts)),[1,8])
        for options in (dict(ragged=1),dict(ragged=True,max_context=8),dict(ragged=False)):
            context=options.pop('max_context',9)
            with self.assertRaises(ValueError):prepare_batch(text,[[1],[1]*8],[8,1],context,**options)

    def test_later_cohort_failure_or_early_close_emits_no_success_and_releases(self):
        for family,path,text,_,batch in self.families():
            model=family.load_reference_model(path);(model.head['head.weight'] if isinstance(model.head,dict) else model.head).zero_()
            prefill=model.prefill;states=[]
            def failing(*args,**kwargs):
                if states:
                    self.assertTrue(states[-1].closed)
                    raise RuntimeError('later cohort failed')
                logits,state=prefill(*args,**kwargs);states.append(state);return logits,state
            rows=[[dict(role='user',content=s)] for s in ('Hi','longer input')]
            runner=batch(model,text,max_context=128);events=[]
            with patch.object(model,'prefill',side_effect=failing):
                with self.assertRaisesRegex(RuntimeError,'later cohort'):
                    for event in runner.generate_batch(rows,max_new_tokens=1,ragged=True):events.append(event)
            self.assertTrue(states[0].closed);self.assertFalse(any('finish_reason' in e for e in events))
            states.clear()
            with patch.object(model,'prefill',side_effect=failing):
                iterator=runner.generate_batch(rows,max_new_tokens=3,ragged=True)
                next(iterator);iterator.close()
            self.assertEqual(len(states),1);self.assertTrue(states[0].closed)
