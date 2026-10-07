import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.deepseek_checkpoint_pipeline import load_reference_pipeline
from dllm.deepseek_reference_loader import load_reference_model
from dllm.deepseek_checkpoint import tensor_specs
from dllm.deepseek_pipeline_generation import DeepSeekPipelineReferenceGenerator
from dllm.deepseek_batch_generation import BatchReferenceGenerator
from dllm.deepseek_generation import ReferenceGenerator
from dllm.deepseek_text import DeepSeekText
from dllm.safetensor_header import ITEM_BYTES
from dllm.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests.test_deepseek_reference_loader import fixture
from tests import test_serve_cpu_reference as serving
from tests.test_deepseek_batch_tools import completion


class DeepSeekLayerPipelineTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,self.headers=fixture(self.path)

    def test_global_weight_admission_and_reordered_decode_match_complete_stack(self):
        specs=tensor_specs(self.config)
        retained=sum(math.prod(s.shape)*(4 if s.storage in ('floating','integer') else 1) for s in specs.values())
        largest=max(math.prod(s.shape)*ITEM_BYTES[self.headers[n]['dtype']] for n,s in specs.items())
        with patch('dllm.deepseek_reference_loader._payload',side_effect=AssertionError('payload')):
            for cuts,budget in (([1,2],retained+largest-1),([3],retained+largest),([1,1],retained+largest),([2,1],retained+largest),([True],retained+largest)):
                with self.assertRaises(ValueError):load_reference_pipeline(self.path,cuts,max_weight_bytes=budget)
        pipeline=load_reference_pipeline(self.path,[1,2],max_weight_bytes=retained+largest);self.addCleanup(pipeline.cleanup)
        self.assertEqual(pipeline.retained_weight_bytes,retained);self.assertEqual(pipeline.weight_overlap_bytes,largest)
        control=load_reference_model(self.path);tokens=torch.arange(18).reshape(2,9)%16
        actual,state=pipeline.prefill(tokens[:,:7],max_tokens=10);self.addCleanup(state.close)
        expected,other=control.prefill(tokens[:,:7],max_tokens=10);other.close()
        torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        order=torch.tensor([1,0,1]);state.reorder(order);tokens=tokens[order]
        for position in (7,8):
            actual=pipeline.decode(tokens[:,position:position+1],state,position=position)
            expected,other=control.prefill(tokens[:,:position+1],max_tokens=10);other.close()
            torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)

    def test_aggregate_cache_and_final_boundary_preflight_and_partial_failure_ownership(self):
        pipeline=load_reference_pipeline(self.path,[1,2]);self.addCleanup(pipeline.cleanup);tokens=torch.tensor([[1,2]])
        reserved=pipeline.cache_bytes_per_row(8)
        with patch.object(pipeline.stages[0],'prefill',side_effect=AssertionError('allocate')):
            with self.assertRaisesRegex(ValueError,'aggregate'):pipeline.prefill(tokens,max_tokens=8,max_cache_bytes=reserved-1)
            last=pipeline.stages[-1];limit=last.max_output_bytes;last.max_output_bytes=4
            with self.assertRaisesRegex(ValueError,'budget'):pipeline.prefill(tokens,max_tokens=8)
            last.max_output_bytes=limit
        _,state=pipeline.prefill(tokens,max_tokens=8,max_cache_bytes=reserved);self.addCleanup(state.close)
        with self.assertRaises(ValueError):state.reorder(torch.tensor([0,0]))
        with patch.object(pipeline.stages[1],'decode',side_effect=RuntimeError('middle')):
            with self.assertRaisesRegex(RuntimeError,'middle'):pipeline.decode(torch.tensor([[3]]),state,position=2)
        self.assertTrue(state.poisoned);self.assertTrue(all(s.poisoned for s in state.stages))
        from dllm.deepseek_layer_stage import DeepSeekStageState
        with patch.object(pipeline.stages[1],'prefill',side_effect=RuntimeError('later')),patch.object(DeepSeekStageState,'close',side_effect=OSError('release')):
            with self.assertRaisesRegex(RuntimeError,'later'):pipeline.prefill(tokens,max_tokens=8)
            self.assertTrue(pipeline.cleanup_pending)
            with patch.object(pipeline.stages[0],'prefill',side_effect=AssertionError('allocate')):
                with self.assertRaisesRegex(OSError,'release'):pipeline.prefill(tokens,max_tokens=8)
        pipeline.cleanup();self.assertFalse(pipeline.cleanup_pending)

    def test_source_receipts_cover_configuration_shards_index_and_final_construction(self):
        import dllm.deepseek_checkpoint_pipeline as builder
        for name,target in (('config.json','load_reference_stage'),('model.safetensors','load_reference_stage'),
                            ('model.safetensors.index.json','load_reference_stage'),('model.safetensors','DeepSeekLayerPipeline')):
            original=getattr(builder,target);calls=[]
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs);calls.append(result)
                source=self.path/name;replacement=self.path/'replacement';replacement.write_bytes(source.read_bytes());replacement.replace(source)
                return result
            with patch.object(builder,target,mutate):
                with self.assertRaisesRegex(ValueError,'changed'):load_reference_pipeline(self.path,[1,2])
            self.assertEqual(len(calls),1)

    def native(self):
        path=serving.ReferenceCommandTests.snapshot(self,'deepseek');pipeline=load_reference_pipeline(path,[1,2]);self.addCleanup(pipeline.cleanup)
        text=DeepSeekText(path,vocab_size=pipeline.embedding.shape[0])
        runner=DeepSeekPipelineReferenceGenerator(pipeline,text,max_context=2048);self.addCleanup(runner.cleanup)
        return runner,BatchReferenceGenerator(load_reference_model(path),text,max_context=2048)

    def test_native_single_equal_ragged_cleanup_and_exact_admission(self):
        runner,control=self.native();message=[dict(role='user',content='Hi')]
        self.assertEqual(list(runner.generate(message,max_new_tokens=3)),list(control.generate(message,max_new_tokens=3)))
        for rows,ragged in (([message]*2,False),([message,[dict(role='user',content='Longer request')]],True)):
            self.assertEqual(list(runner.generate_batch(rows,max_new_tokens=[1,3],ragged=ragged)),list(control.generate_batch(rows,max_new_tokens=[1,3],ragged=ragged)))
        with self.assertRaises(ValueError):ReferenceGenerator(runner.model,runner.text)
        with self.assertRaises(ValueError):DeepSeekPipelineReferenceGenerator(runner.model.stages[0],runner.text)
        stream=runner.generate(message,max_new_tokens=8);next(stream);stream.close()
        stream=runner.generate(message,max_new_tokens=8);next(stream)
        with cancellation_scope(lambda:True):
            with self.assertRaises(ReferenceCancelled):next(stream)
        self.assertTrue(list(runner.generate(message,max_new_tokens=1)))

    def test_native_dsml_after_real_pipeline_execution(self):
        runner,_=self.native();tokens=runner.text.tokenizer.encode(completion(),add_special_tokens=False).ids
        prefill,decode=runner.model.prefill,runner.model.decode;step=0
        def choose(logits):
            nonlocal step
            logits.zero_();logits[:,tokens[step]]=10;step+=1;return logits
        def start(*args,**kwargs):
            nonlocal step
            step=0;logits,state=prefill(*args,**kwargs);return choose(logits),state
        def advance(*args,**kwargs):return choose(decode(*args,**kwargs))
        tools=[dict(type='function',function=dict(name='lookup',strict=True,parameters=dict(type='object',properties={'value':{'type':'string'}},required=['value'],additionalProperties=False)))]
        with patch.object(runner.model,'prefill',start),patch.object(runner.model,'decode',advance):
            events=list(runner.generate([dict(role='user',content='Hi')],tools=tools,max_new_tokens=len(tokens)))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        self.assertEqual(next(e['tool_calls'] for e in events if 'tool_calls' in e)[0]['function']['name'],'lookup')
