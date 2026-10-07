import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_layer_stage import FlashLayerStage
from dllm.glm_flash_reference_loader import load_reference_stage
from dllm.glm_flash_model import FlashModel
from dllm.glm_flash_checkpoint import tensor_specs,ROOT
from dllm.safetensor_header import ITEM_BYTES
from tests.test_glm_flash_reference_loader import fixture,floating_control


class FlashLayerStageTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,self.headers,self.values=fixture(self.path)
        self.control=floating_control(self.config,self.values)

    def stages(self,ranges,**options):
        stages=[load_reference_stage(self.path,a,b,**options) for a,b in ranges]
        for stage in stages:self.addCleanup(stage.cleanup)
        return stages

    def state(self,stage,batch=2,capacity=6):
        state=stage.new_state(batch,capacity);self.addCleanup(state.close);return state

    def test_two_three_chunked_stages_preserve_streams_and_independent_logits(self):
        tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
        expected,control_state=self.control.prefill(tokens,capacity=6,all_logits=True);self.addCleanup(control_state.close)
        for ranges in ([(0,1),(1,3)],[(0,2),(2,3)],[(0,1),(1,2),(2,3)]):
            stages=self.stages(ranges);states=[self.state(stage) for stage in stages]
            self.assertTrue(all(not isinstance(stage,FlashModel) for stage in stages))
            for position,length in ((0,2),(2,1),(3,1)):
                hidden=tokens[:,position:position+length]
                for i,(stage,state) in enumerate(zip(stages,states)):
                    hidden=stage.forward(hidden,state,position=position,all_logits=stage.last)
                    if not stage.last:self.assertEqual(tuple(hidden.shape),(2,length,stage.streams,stage.dim))
                torch.testing.assert_close(hidden,expected[:,position:position+length],rtol=3e-5,atol=3e-6)
            order=torch.tensor([1,0,1])
            for state in states:state.reorder(order)
            next_tokens=torch.tensor([[5],[6],[7]])
            reference=floating_control(self.config,self.values)
            history=torch.cat((tokens[order],next_tokens),1)
            wanted,state=reference.prefill(history,capacity=6);state.close()
            hidden=next_tokens
            for stage,state in zip(stages,states):hidden=stage.forward(hidden,state,position=4)
            torch.testing.assert_close(hidden,wanted,rtol=3e-5,atol=3e-6)

    def test_range_payload_selection_and_exact_selected_weight_budget(self):
        import dllm.glm_flash_reference_loader as loader
        specs=tensor_specs(self.config)
        for start,end in ((0,1),(1,2),(2,3)):
            names={name for name in specs if name.startswith(tuple(ROOT+f'layers.{i}.' for i in range(start,end)))}
            if start==0:names.add(ROOT+'embed_tokens.weight')
            if end==3:names|={ROOT+'norm.weight','lm_head.weight'}
            retained=sum(math.prod(specs[n].shape)*(1 if specs[n].storage=='fp8' else 4) for n in names)
            largest=max(math.prod(specs[n].shape)*ITEM_BYTES[self.headers[n]['dtype']] for n in names)
            with patch.object(loader,'_payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_reference_stage(self.path,start,end,max_weight_bytes=retained+largest-1)
            original=loader._payload
            with patch.object(loader,'_payload',wraps=original) as reads:
                stage=load_reference_stage(self.path,start,end,max_weight_bytes=retained+largest)
            self.assertEqual(reads.call_count,len(names));self.assertEqual((stage.start,stage.end),(start,end))
            self.assertEqual(stage.embedding is not None,start==0);self.assertEqual(stage.head is not None,end==3)
            for block in stage.blocks:
                if block.feed_forward.experts:self.assertEqual(block.feed_forward.experts[0].gate.weight.dtype,torch.uint8)

    def test_bad_ranges_full_header_audit_and_output_preflight_precede_mutation(self):
        import dllm.glm_flash_reference_loader as loader
        with patch.object(loader,'_payload',side_effect=AssertionError('payload')):
            for start,end in ((-1,2),(0,0),(0,4),(True,2),(None,2),(1,None)):
                with self.assertRaises(ValueError):load_reference_stage(self.path,start,end)
            with patch.object(loader,'audit_headers',side_effect=ValueError('full audit')):
                with self.assertRaisesRegex(ValueError,'full audit'):load_reference_stage(self.path,0,1)
        first,last=self.stages([(0,1),(1,3)]);state=self.state(last)
        for hidden in (torch.zeros(2,1,16),torch.zeros(2,1,1,16),torch.full((2,1,2,16),float('nan'))):
            with self.assertRaises(ValueError):last.forward(hidden,state,position=0)
        with self.assertRaises(ValueError):last.forward(torch.zeros(2,1,2,16),state,position=1)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)
        limited=FlashLayerStage(first.blocks,start=0,total=3,embedding=first.embedding,max_output_bytes=4)
        state=self.state(limited)
        with self.assertRaisesRegex(ValueError,'budget'):limited.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)

    def test_partial_layer_failure_poisoning_and_retained_cleanup_retry(self):
        stage=self.stages([(0,3)])[0];state=self.state(stage)
        with patch.object(stage.blocks[1],'forward',side_effect=RuntimeError('later layer')):
            with self.assertRaisesRegex(RuntimeError,'later layer'):stage.forward(torch.tensor([[1],[2]]),state,position=0)
        self.assertTrue(state.poisoned)
        with self.assertRaises(RuntimeError):stage.forward(torch.tensor([[1],[2]]),state,position=0)
        state.close()
        first=stage.blocks[0];original=first.new_state;created=[]
        def allocate(*args,**kwargs):
            value=original(*args,**kwargs);created.append(value);return value
        with patch.object(first,'new_state',allocate),patch.object(stage.blocks[1],'new_state',side_effect=MemoryError('allocation')):
            from dllm.glm_flash_block import FlashBlockState
            with patch.object(FlashBlockState,'close',side_effect=RuntimeError('cleanup')):
                with self.assertRaises(MemoryError):stage.new_state(1,4)
                self.assertTrue(stage.failed_cleanups)
                with self.assertRaisesRegex(RuntimeError,'cleanup'):stage.new_state(1,4)
        stage.cleanup();self.assertFalse(stage.failed_cleanups);self.assertTrue(created[0].attention.closed);self.assertIsNone(created[0].owner)
        self.assertIsNotNone(self.state(stage))

    def test_stage_prefill_cache_budget_and_unowned_header_rejection(self):
        first,last=self.stages([(0,1),(1,3)]);tokens=torch.tensor([[1,2],[3,4]])
        hidden,left=first.prefill(tokens,capacity=4);self.addCleanup(left.close)
        logits,right=last.prefill(hidden,capacity=4,all_logits=True);self.addCleanup(right.close)
        expected,state=self.control.prefill(tokens,capacity=4,all_logits=True);state.close()
        torch.testing.assert_close(logits,expected,rtol=3e-5,atol=3e-6)
        with patch.object(last.blocks[0],'new_state',side_effect=AssertionError('allocate')):
            with self.assertRaisesRegex(ValueError,'aggregate'):
                last.new_state(2,4,max_cache_bytes=2*last._cache_per_slot(4)-1)
        # A range cannot hide an invalid required tensor owned by another stage.
        import json
        from dllm.safetensor_header import read_header
        from tests.test_packed_checkpoint import write_raw
        name=ROOT+'layers.2.input_layernorm.weight'
        mapping=json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
        file=self.path/mapping[name];raw=file.read_bytes()
        with file.open('rb') as reader:entries,base,_=read_header(reader)
        entries[name]['dtype']='I16';write_raw(file,entries,raw[base:])
        with patch('dllm.glm_flash_reference_loader._payload',side_effect=AssertionError('payload')):
            with self.assertRaises(ValueError):load_reference_stage(self.path,0,1)
