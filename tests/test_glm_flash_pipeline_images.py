import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_checkpoint_pipeline import load_reference_pipeline
from dllm.glm_flash_reference_loader import load_reference_model
from dllm.glm_flash_image_embeddings import image_embeddings
from dllm.glm_flash_pipeline_image_generation import FlashPipelineImageReferenceGenerator,FlashPipelineImageToolReferenceGenerator
from dllm.glm_flash_image_generation import FlashImageReferenceGenerator
from dllm.glm_flash_image_text import FlashImageText
from dllm.glm_flash_image_tool_text import FlashImageToolText
from dllm.glm_flash_vision_loader import load_reference_vision
from dllm.glm_flash_inline_images import decode_inline_images
from dllm.glm_flash_tools import FlashTools
from dllm.glm_text import END_THINK,EOS
from dllm.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests.test_glm_flash_reference_loader import fixture,floating_control
from tests.test_glm_flash_inline_images import snapshot,messages
from tests.test_glm_tools import TOOLS,CALLS


class FlashPipelineImageTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def test_prepared_feature_chunks_reorder_then_token_decode_match_independent_control(self):
        config,_,weights=fixture(self.path);control=floating_control(config,weights)
        pipeline=load_reference_pipeline(self.path,[1,2]);self.addCleanup(pipeline.cleanup)
        values=torch.randn(2,3,16,generator=torch.Generator().manual_seed(11))*.1;before=values.clone()
        actual,state=pipeline.prefill_embeddings(values[:,:2],capacity=5,all_logits=True);self.addCleanup(state.close)
        expected,other=control.prefill_embeddings(values[:,:2],capacity=5,all_logits=True);other.close()
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        order=torch.tensor([1,0,1]);state.reorder(order)
        actual=pipeline.forward_embeddings(values[order,2:],state,position=2)
        expected,other=control.prefill_embeddings(values[order],capacity=5);other.close()
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        tokens=torch.tensor([[3],[4],[5]]);actual=pipeline.forward(tokens,state,position=3)
        history=torch.cat((values[order],torch.nn.functional.embedding(tokens,control.embedding)),1)
        expected,other=control.prefill_embeddings(history,capacity=5);other.close()
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6);self.assertTrue(torch.equal(values,before))

    def test_invalid_features_boundary_budget_and_placeholder_counts_precede_execution(self):
        fixture(self.path);pipeline=load_reference_pipeline(self.path,[1,2]);state=pipeline.new_state(1,4);self.addCleanup(state.close)
        good=torch.zeros(1,2,16)
        with patch.object(pipeline.stages[0],'forward_embeddings',side_effect=AssertionError('execute')):
            for values in (good.double(),good[:,:,None,:],good[:,:,:8],torch.full_like(good,float('nan'))):
                with self.assertRaises(ValueError):pipeline.forward_embeddings(values,state,position=0)
            pipeline.stages[-1].max_output_bytes=4
            with self.assertRaisesRegex(ValueError,'budget'):pipeline.forward_embeddings(good,state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)
        with self.assertRaises(ValueError):pipeline.stages[1].forward_embeddings(good,state,position=0)
        pipeline.stages[-1].max_output_bytes=64*1024**2
        tokens=torch.tensor([[1,0,2]]);features=[torch.ones(1,16)]
        values=image_embeddings(pipeline,tokens,features,image_token_id=0)
        self.assertTrue(torch.equal(values[0,1],features[0][0]));self.assertEqual(tokens.tolist(),[[1,0,2]])
        with self.assertRaises(ValueError):image_embeddings(pipeline,tokens,[torch.zeros(2,16)],image_token_id=0)
        with self.assertRaises(ValueError):image_embeddings(pipeline.stages[0],tokens,features,image_token_id=0)

    def prepare(self,tools=False):
        snapshot(self.path);pipeline=load_reference_pipeline(self.path,[1,2]);self.addCleanup(pipeline.cleanup)
        text=(FlashImageToolText if tools else FlashImageText)(self.path,vocab_size=pipeline.embedding.shape[0])
        runner=(FlashPipelineImageToolReferenceGenerator if tools else FlashPipelineImageReferenceGenerator)(
            pipeline,text,load_reference_vision(self.path),max_context=2048)
        self.addCleanup(runner.cleanup);decoded=decode_inline_images(messages())
        return runner,decoded

    def test_real_packed_vision_language_matches_whole_model_and_cleanup_reuse(self):
        runner,decoded=self.prepare();other=FlashImageReferenceGenerator(load_reference_model(self.path),runner.text,load_reference_vision(self.path),max_context=2048)
        expected=list(other.generate_images(decoded.messages,decoded.images,max_new_tokens=3))
        self.assertEqual(list(runner.generate_images(decoded.messages,decoded.images,max_new_tokens=3)),expected)
        stream=runner.generate_images(decoded.messages,decoded.images,max_new_tokens=8);next(stream);stream.close()
        stream=runner.generate_images(decoded.messages,decoded.images,max_new_tokens=8);next(stream)
        with cancellation_scope(lambda:True):
            with self.assertRaises(ReferenceCancelled):next(stream)
        self.assertEqual(list(runner.generate_images(decoded.messages,decoded.images,max_new_tokens=3)),expected)

    def test_native_image_tools_after_real_vision_and_pipeline_execution(self):
        runner,decoded=self.prepare(tools=True);tokens=runner.text.tokenizer.encode(END_THINK+FlashTools(TOOLS).render(CALLS)+EOS,add_special_tokens=False).ids
        original=runner.model._forward;step=0
        def select(*args,**kwargs):
            nonlocal step
            if kwargs['position']==0:step=0
            logits=original(*args,**kwargs);logits.zero_();logits[:,tokens[step]]=10;step+=1;return logits
        with patch.object(runner.model,'_forward',select):events=list(runner.generate_image_tools(decoded.messages,decoded.images,TOOLS,max_new_tokens=len(tokens)))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        call=next(e['tool_calls'] for e in events if 'tool_calls' in e)[0]['function']
        self.assertEqual(call['name'],CALLS[0]['function']['name'])
        self.assertEqual(json.loads(call['arguments']),json.loads(CALLS[0]['function']['arguments']))

    def test_retained_cache_failure_blocks_new_vision_until_retry(self):
        runner,decoded=self.prepare();states=[];create=runner.model.new_state
        def observe(*args,**kwargs):
            state=create(*args,**kwargs);states.append(state);return state
        with patch.object(runner.model,'new_state',observe):
            stream=runner.generate_images(decoded.messages,decoded.images,max_new_tokens=3);next(stream);state=states[-1]
            with patch.object(state.stages[0],'close',side_effect=RuntimeError('release')):
                with self.assertRaisesRegex(RuntimeError,'release'):list(stream)
                with patch.object(runner.vision,'forward',side_effect=AssertionError('vision')):
                    with self.assertRaisesRegex(RuntimeError,'release'):list(runner.generate_images(decoded.messages,decoded.images,max_new_tokens=1))
            runner.cleanup();self.assertTrue(state.closed)
        self.assertTrue(list(runner.generate_images(decoded.messages,decoded.images,max_new_tokens=1)))
