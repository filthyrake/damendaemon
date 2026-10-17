from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from damendaemon.glm_flash_tools import FlashTools
from damendaemon.glm_text import END_THINK,EOS
from tests import test_glm_tools as glm
from tests import test_glm_flash_batch_tools as flash
from tests import test_deepseek_batch_tools as deepseek
from tests.test_glm_flash_tool_generation import TOOLS


class RaggedToolBatchTests(unittest.TestCase):
    def cases(self,*,malformed=False):
        for family in (glm,flash,deepseek):
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)
            base=family.text_fixture(path)
            if family is glm:
                text=family.GlmToolText(path,vocab_size=base.vocab_size);model_type=family.ScriptedGlm;runner_type=family.GlmToolReferenceGenerator
                family.fixture(path,vocab_size=text.vocab_size)
            elif family is flash:
                text=family.FlashToolText(path,vocab_size=base.vocab_size);model_type=family.ScriptedBatch;runner_type=family.FlashToolReferenceGenerator
                family.snapshot(path,text)
            else:
                text=base;model_type=family.ScriptedDeepSeek;runner_type=family.BatchReferenceGenerator
                family.fixture(path,vocab_size=text.vocab_size)
            def call(value):
                if family is deepseek:return deepseek.completion(value=value)
                wire=FlashTools(TOOLS).render([dict(type='function',function=dict(name='lookup',arguments=json.dumps({'value':value})))])
                return END_THINK+wire+EOS
            plain=('' if family is deepseek else END_THINK)+'answer'+family.EOS
            completions=[call('first'),call('bad').replace('lookup','unknown') if malformed else plain,call('second')]
            model=model_type(text,completions);runner=runner_type(model,text,max_context=4096)
            rows=[[dict(role='user',content=s)] for s in ('Hi','longer input','Yo')]
            if family is deepseek:
                generate=lambda **options:runner.generate_batch(rows,tools=TOOLS,max_new_tokens=512,**options)
                prompts=[text.chat_tokens(row,tools=TOOLS) for row in rows]
            else:
                generate=lambda **options:runner.generate_batch_tools(rows,TOOLS,max_new_tokens=512,**options)
                prompts=[text.tool_chat_tokens(row,TOOLS) for row in rows]
            self.assertEqual(len(prompts[0]),len(prompts[2]));self.assertNotEqual(len(prompts[0]),len(prompts[1]))
            prefill=model.prefill;cohorts=iter(([0,2],[1]));seen=[]
            def grouped_prefill(tokens,**options):
                if model.states:self.assertTrue(model.states[-1].closed)
                active=next(cohorts);seen.append(active)
                self.assertEqual(tokens.tolist(),[prompts[row] for row in active])
                _,state=prefill(tokens,**options);state.rows=list(active)
                return model.logits(state),state
            yield SimpleNamespace(family=family,path=path,text=text,model=model,runner=runner,rows=rows,
                                  generate=generate,prefill=grouped_prefill,seen=seen,prompts=prompts)

    def test_mixed_tool_replies_preserve_rows_usage_and_stable_cohorts(self):
        for case in self.cases():
            with self.subTest(family=case.family.__name__),patch.object(case.model,'prefill',side_effect=case.prefill):
                events=list(case.generate(ragged=True))
            self.assertEqual(case.seen,[[0,2],[1]])
            finals=[e for e in events if 'finish_reason' in e]
            self.assertEqual([e['row'] for e in finals],[0,1,2])
            self.assertEqual([e['finish_reason'] for e in finals],['tool_calls','stop','tool_calls'])
            self.assertEqual([e['usage']['prompt_tokens'] for e in finals],list(map(len,case.prompts)))
            emitted={e['row']:e['tool_calls'] for e in events if 'tool_calls' in e}
            self.assertEqual({row:json.loads(calls[0]['function']['arguments']) for row,calls in emitted.items()},
                             {0:{'value':'first'},2:{'value':'second'}})
            self.assertEqual(''.join(e.get('content','') for e in events if e['row']==1),'answer')
            self.assertTrue(all(s.closed for s in case.model.states))

    def test_later_malformed_call_suppresses_all_success_finals(self):
        for case in self.cases(malformed=True):
            events=[]
            with patch.object(case.model,'prefill',side_effect=case.prefill),self.assertRaises(ValueError):
                for event in case.generate(ragged=True):events.append(event)
            self.assertFalse(any('finish_reason' in e for e in events))
            self.assertFalse(any(e.get('tool_calls') and e['row']==1 for e in events))
            self.assertTrue(all(s.closed for s in case.model.states))

    def test_abandonment_owns_all_glm_call_buffers_and_retryable_cache_cleanup(self):
        for case in self.cases():
            streams=[]
            if case.family is not deepseek:
                original=case.runner.stream_type
                def retained(*args,**kwargs):
                    stream=original(*args,**kwargs);streams.append(stream);return stream
                case.runner.stream_type=retained
            with patch.object(case.model,'prefill',side_effect=case.prefill):
                stream=case.generate(ragged=True);next(stream);state=case.model.states[0]
                with patch.object(state,'close',side_effect=RuntimeError('cleanup')):
                    stream.close()
                    with patch.object(case.model,'prefill',side_effect=AssertionError('new prefill')):
                        with self.assertRaisesRegex(RuntimeError,'cleanup'):list(case.generate(ragged=True))
                case.runner.cleanup()
            self.assertTrue(state.closed);self.assertEqual(len(case.model.states),1)
            if case.family is not deepseek:
                self.assertGreaterEqual(len(streams),3)
                self.assertTrue(all(s.parser.closed for s in streams))

    def test_cli_tool_ragged_forwarding_and_bad_later_history_precedes_load(self):
        for case in self.cases():
            messages=case.path/'batch.json';messages.write_text(json.dumps(case.rows))
            tools=case.path/'tools.json';tools.write_text(json.dumps(TOOLS))
            argv=[str(case.path),'--batch-messages',str(messages),'--tools',str(tools),'--ragged-batch','--max-new-tokens','512']
            output=io.StringIO()
            with patch.object(case.family.cli,'load_reference_model',return_value=case.model),\
                 patch.object(case.model,'prefill',side_effect=case.prefill),redirect_stdout(output):
                self.assertEqual(case.family.cli.main(argv),0)
            events=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([e['finish_reason'] for e in events if 'finish_reason' in e],['tool_calls','stop','tool_calls'])
            messages.write_text(json.dumps(case.rows[:1]+[[dict(role='tool',tool_call_id='foreign',content='bad')]]))
            with patch.object(case.family.cli,'load_reference_model',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:case.family.cli.main(argv)
                self.assertEqual(error.exception.code,2)
