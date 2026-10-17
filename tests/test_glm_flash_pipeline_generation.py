import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from damendaemon.glm_flash_layer_pipeline import FlashLayerPipeline
from damendaemon.glm_flash_pipeline_generation import FlashPipelineReferenceGenerator,FlashPipelineToolReferenceGenerator
from damendaemon.glm_flash_reference_loader import load_reference_stage
from damendaemon.glm_flash_batch_generation import FlashBatchReferenceGenerator
from damendaemon.glm_flash_generation import FlashReferenceGenerator
from damendaemon.glm_flash_tool_text import FlashToolText
from damendaemon.glm_text import END_THINK,EOS
from damendaemon.glm_flash_tools import FlashTools
from damendaemon.reference_serving import ReferenceRuntime
from damendaemon.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests.test_glm_flash_generation import text_fixture
from tests.test_glm_flash_reference_loader import fixture,floating_control
from tests.test_glm_tools import TOOLS,CALLS


class FlashPipelineGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.text=text_fixture(self.path);config,_,values=fixture(self.path,vocab_size=self.text.vocab_size)
        config['text_config'].update(eos_token_id=sorted(self.text.eos),pad_token_id=self.text.tokenizer.token_to_id(EOS),bos_token_id=None)
        (self.path/'config.json').write_text(json.dumps(config))
        self.pipeline=FlashLayerPipeline([load_reference_stage(self.path,0,1),load_reference_stage(self.path,1,3)])
        self.addCleanup(self.pipeline.cleanup);self.control=floating_control(config,values)
        self.messages=[dict(role='user',content='Hi')];self.states=[];create=self.pipeline.new_state
        def observe(*args,**kwargs):
            state=create(*args,**kwargs);self.states.append(state);return state
        watcher=patch.object(self.pipeline,'new_state',observe);watcher.start();self.addCleanup(watcher.stop)

    def test_native_text_single_equal_ragged_control_and_admission(self):
        runner=FlashPipelineReferenceGenerator(self.pipeline,self.text,max_context=1024)
        other=FlashBatchReferenceGenerator(self.control,self.text,max_context=1024)
        self.assertEqual(list(runner.generate(self.messages,max_new_tokens=3)),list(other.generate(self.messages,max_new_tokens=3)))
        for rows,ragged in (([self.messages]*2,False),([self.messages,[dict(role='user',content='Longer prompt')]],True)):
            self.assertEqual(list(runner.generate_batch(rows,max_new_tokens=[1,3],ragged=ragged)),list(other.generate_batch(rows,max_new_tokens=[1,3],ragged=ragged)))
        self.assertTrue(all(state.closed for state in self.states))
        with self.assertRaises(ValueError):FlashReferenceGenerator(self.pipeline,self.text,max_context=1024)
        with self.assertRaises(ValueError):ReferenceRuntime(runner)
        with self.assertRaises(ValueError):FlashPipelineReferenceGenerator(self.pipeline.stages[0],self.text,max_context=1024)

    def test_native_tools_over_real_pipeline_and_ragged_rows(self):
        text=FlashToolText(self.path,vocab_size=self.text.vocab_size)
        runner=FlashPipelineToolReferenceGenerator(self.pipeline,text,max_context=4096)
        tokens=text.tokenizer.encode(END_THINK+FlashTools(TOOLS).render(CALLS)+EOS,add_special_tokens=False).ids
        original=self.pipeline.forward;step=0
        def select(inputs,state,**kwargs):
            nonlocal step
            if kwargs['position']==0:step=0
            logits=original(inputs,state,**kwargs);logits.zero_();logits[:,tokens[step]]=10;step+=1;return logits
        with patch.object(self.pipeline,'forward',select):
            events=list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=len(tokens)))
            rows=list(runner.generate_batch_tools([self.messages,[dict(role='user',content='Longer request')]],TOOLS,max_new_tokens=len(tokens),ragged=True))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        call=next(e['tool_calls'] for e in events if 'tool_calls' in e)[0]
        self.assertEqual(call['function']['name'],CALLS[0]['function']['name'])
        self.assertEqual(json.loads(call['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        self.assertEqual([e['row'] for e in rows if 'finish_reason' in e],[0,1])
        self.assertTrue(all(state.closed for state in self.states))
        runtime=ReferenceRuntime(runner);self.addCleanup(runtime.close)
        self.assertEqual(runtime.family,'glm5_next')
        self.assertEqual(runtime.execution_status()['stage_layers'],[[0,1],[1,3]])

    def test_abandonment_cancellation_and_failed_cleanup_before_reuse(self):
        runner=FlashPipelineReferenceGenerator(self.pipeline,self.text,max_context=1024)
        stream=runner.generate(self.messages,max_new_tokens=8);next(stream);stream.close();self.assertTrue(self.states[-1].closed)
        stream=runner.generate(self.messages,max_new_tokens=8);next(stream)
        with cancellation_scope(lambda:True):
            with self.assertRaises(ReferenceCancelled):next(stream)
        self.assertTrue(self.states[-1].closed)
        stream=runner.generate(self.messages,max_new_tokens=4);next(stream);state=self.states[-1]
        with patch.object(state.stages[0],'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'release'):list(stream)
            count=len(self.states)
            with self.assertRaisesRegex(RuntimeError,'release'):list(runner.generate(self.messages,max_new_tokens=1))
            self.assertEqual(len(self.states),count)
        runner.cleanup();self.assertTrue(state.closed);self.assertTrue(list(runner.generate(self.messages,max_new_tokens=1)))
