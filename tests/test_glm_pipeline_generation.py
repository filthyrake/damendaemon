import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_layer_pipeline import GlmLayerPipeline
from dllm.glm_pipeline_generation import PipelineReferenceGenerator, PipelineToolReferenceGenerator
from dllm.glm_reference_loader import load_reference_stage
from dllm.glm_batch_generation import BatchReferenceGenerator
from dllm.glm_generation import ReferenceGenerator
from dllm.glm_tool_text import GlmToolText
from dllm.glm_text import END_THINK, EOS
from dllm.glm_flash_tools import FlashTools
from dllm.reference_cancellation import cancellation_scope, ReferenceCancelled
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture, floating_control
from tests.test_glm_tools import TOOLS, CALLS


class GlmPipelineGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path)
        config,_,values=fixture(self.path,vocab_size=self.text.vocab_size)
        self.pipeline=GlmLayerPipeline([load_reference_stage(self.path,0,2),load_reference_stage(self.path,2,3)])
        self.control=floating_control(config,values)
        self.messages=[dict(role='user',content='Hi')]
        self.states=[];create=self.pipeline.new_state
        def observe(*args,**kwargs):
            state=create(*args,**kwargs);self.states.append(state);return state
        watcher=patch.object(self.pipeline,'new_state',observe);watcher.start();self.addCleanup(watcher.stop)
        self.addCleanup(self.pipeline.cleanup)

    def runner(self):return PipelineReferenceGenerator(self.pipeline,self.text,max_context=1024)

    def test_real_packed_single_and_ragged_generation_matches_whole_model(self):
        runner=self.runner();other=BatchReferenceGenerator(self.control,self.text,max_context=1024)
        self.assertEqual(list(runner.generate(self.messages,max_new_tokens=3)),list(other.generate(self.messages,max_new_tokens=3)))
        rows=[self.messages,[dict(role='user',content='A longer prompt')],self.messages]
        actual=list(runner.generate_batch(rows,max_new_tokens=[1,3,2],ragged=True))
        expected=list(other.generate_batch(rows,max_new_tokens=[1,3,2],ragged=True))
        self.assertEqual(actual,expected)
        self.assertTrue(all(state.closed for state in self.states))
        self.assertEqual(len([event for event in actual if 'finish_reason' in event]),3)
        with self.assertRaises(ValueError):ReferenceGenerator(self.pipeline,self.text,max_context=1024)
        with self.assertRaises(ValueError):PipelineReferenceGenerator(self.pipeline.stages[0],self.text,max_context=1024)

    def test_abandonment_cancellation_and_later_failure_release_whole_owner(self):
        runner=self.runner()
        stream=runner.generate(self.messages,max_new_tokens=5);next(stream);stream.close()
        self.assertTrue(self.states[-1].closed)
        stream=runner.generate(self.messages,max_new_tokens=5);next(stream)
        with cancellation_scope(lambda:True):
            with self.assertRaises(ReferenceCancelled):next(stream)
        self.assertTrue(self.states[-1].closed)
        stream=runner.generate(self.messages,max_new_tokens=5);next(stream)
        with patch.object(self.pipeline.stages[-1],'forward',side_effect=RuntimeError('last stage')):
            with self.assertRaisesRegex(RuntimeError,'last stage'):list(stream)
        self.assertTrue(self.states[-1].closed)
        self.assertTrue(list(runner.generate(self.messages,max_new_tokens=1)))

    def test_failed_release_prevents_success_and_new_allocation_until_retry(self):
        runner=self.runner();stream=runner.generate(self.messages,max_new_tokens=3);next(stream)
        state=self.states[-1];local=state.stages[0]
        with patch.object(local,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'release'):list(stream)
            count=len(self.states)
            with self.assertRaisesRegex(RuntimeError,'release'):list(runner.generate(self.messages,max_new_tokens=1))
            self.assertEqual(len(self.states),count)
        runner.cleanup();self.assertTrue(state.closed);self.assertIsNone(state.owner)
        self.assertTrue(list(runner.generate(self.messages,max_new_tokens=1)))

    def test_native_tool_parser_over_real_pipeline_with_controlled_token_selection(self):
        text=GlmToolText(self.path,vocab_size=self.text.vocab_size)
        runner=PipelineToolReferenceGenerator(self.pipeline,text,max_context=4096)
        wire=END_THINK+FlashTools(TOOLS).render(CALLS)+EOS
        tokens=text.tokenizer.encode(wire,add_special_tokens=False).ids
        original=self.pipeline.forward;step=0
        def select(inputs,state,**kwargs):
            nonlocal step
            if kwargs['position']==0:step=0
            logits=original(inputs,state,**kwargs)
            # Real packed stages execute every step; deterministic selection
            # tests the pinned protocol, not trained-model tool competence.
            logits.zero_();logits[:,tokens[step]]=10;step+=1
            return logits
        with patch.object(self.pipeline,'forward',select):
            events=list(runner.generate_tools(self.messages,TOOLS,max_new_tokens=len(tokens)))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        calls=next(event['tool_calls'] for event in events if 'tool_calls' in event)
        self.assertEqual(calls[0]['function']['name'],CALLS[0]['function']['name'])
        self.assertEqual(json.loads(calls[0]['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        with patch.object(self.pipeline,'forward',select):
            events=list(runner.generate_batch_tools([self.messages,[dict(role='user',content='Longer request')]],TOOLS,
                max_new_tokens=len(tokens),ragged=True))
        self.assertEqual([e['row'] for e in events if 'finish_reason' in e],[0,1])
        self.assertTrue(all(state.closed for state in self.states))
        self.assertTrue(self.states[-1].closed)
        step=0
        stream=runner.generate_tools(self.messages,TOOLS,max_new_tokens=len(tokens))
        with patch.object(self.pipeline,'forward',select),cancellation_scope(lambda:step>=2):
            with self.assertRaises(ReferenceCancelled):list(stream)
        self.assertTrue(self.states[-1].closed)

    def test_context_and_cache_limits_fail_before_stage_execution(self):
        runner=PipelineReferenceGenerator(self.pipeline,self.text,max_context=1)
        with patch.object(self.pipeline,'prefill',side_effect=AssertionError('prefill')):
            with self.assertRaises(ValueError):list(runner.generate(self.messages,max_new_tokens=1))
        runner=PipelineReferenceGenerator(self.pipeline,self.text,max_context=1024,max_cache_bytes=1)
        with patch.object(self.pipeline.stages[0],'forward',side_effect=AssertionError('execute')):
            with self.assertRaisesRegex(ValueError,'aggregate'):list(runner.generate(self.messages,max_new_tokens=1))
