import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_layer_stage import GlmLayerStage
from damendaemon.glm_reference_loader import load_reference_stage
from damendaemon.glm_checkpoint import tensor_specs
from damendaemon.safetensor_header import ITEM_BYTES
from tests.test_glm_model import model_fixture
from tests.test_glm_reference_loader import fixture, floating_control


def prepared(model, ranges):
    return [GlmLayerStage(model.blocks[start:end], start=start, total=len(model.blocks),
        embedding=model.embedding if start==0 else None,
        norm=model.norm if end==len(model.blocks) else None,
        head=model.head if end==len(model.blocks) else None) for start,end in ranges]


def forward(stages, states, tokens, position, all_logits=False):
    hidden=tokens
    for stage,state in zip(stages,states):
        hidden=stage.forward(hidden,state,position=position,all_logits=all_logits and stage.last)
    return hidden


class GlmLayerStageTests(unittest.TestCase):
    def test_prepared_two_and_three_stage_chunks_and_reorder(self):
        for cuts in ((2,), (2,3)):
            model=model_fixture()
            # Append another full-indexer block to provide a three-stage cut.
            if len(cuts)==2:
                other=model_fixture();model.blocks=(*model.blocks,other.blocks[-1])
            boundaries=(0,*cuts,len(model.blocks))
            stages=prepared(model,list(zip(boundaries,boundaries[1:])))
            states=[stage.new_state(2,6) for stage in stages]
            for state in states:self.addCleanup(state.close)
            tokens=torch.tensor([[1,2,3,4,5],[3,2,1,5,4]])
            expected,control=model.prefill(tokens,capacity=6,all_logits=True);self.addCleanup(control.close)
            actual=forward(stages,states,tokens[:,:2],0,True)
            torch.testing.assert_close(actual,expected[:,:2])
            order=torch.tensor([1,0,1])
            for state in states:state.reorder(order)
            actual=forward(stages,states,tokens[order,2:4],2,True)
            torch.testing.assert_close(actual,expected[order,2:4])
            for state in states:state.reorder(torch.tensor([2]))
            actual=forward(stages,states,tokens[1:2,4:5],4)
            torch.testing.assert_close(actual,expected[1:2,4])
            self.assertTrue(all(state.position==5 for state in states))

    def test_packed_stage_payload_selection_budget_and_floating_control(self):
        import damendaemon.glm_reference_loader as loader
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers,values=fixture(path)
            specs=tensor_specs(config);stages=[]
            for start,end in ((0,2),(2,3)):
                names={name for name in specs if name.startswith(tuple(f'model.layers.{i}.' for i in range(start,end)))}
                if start==0:names.add('model.embed_tokens.weight')
                if end==3:names|={'model.norm.weight','lm_head.weight'}
                retained=sum(math.prod(specs[n].shape)*(1 if specs[n].storage=='fp8' else 4) for n in names)
                largest=max(math.prod(specs[n].shape)*ITEM_BYTES[headers[n]['dtype']] for n in names)
                with patch.object(loader,'_payload',side_effect=AssertionError('over-budget read')):
                    with self.assertRaisesRegex(ValueError,'weight budget'):
                        load_reference_stage(path,start,end,max_weight_bytes=retained+largest-1)
                original=loader._payload;seen=[]
                def read(file,base,entry):
                    seen.append((Path(file.name).name,entry['data_offsets']))
                    return original(file,base,entry)
                with patch.object(loader,'_payload',read):
                    stage=load_reference_stage(path,start,end,max_weight_bytes=retained+largest)
                # The generated checkpoint alternates each spec across two files.
                expected={(f'part-{i%2}.safetensors',tuple(headers[n]['data_offsets']))
                          for i,n in enumerate(specs) if n in names}
                self.assertEqual({(file,tuple(offsets)) for file,offsets in seen},expected)
                self.assertEqual(len(seen),len(names))
                self.assertEqual(stage.embedding is not None,start==0)
                self.assertEqual(stage.head is not None,end==3)
                stages.append(stage)
            control=floating_control(config,values)
            tokens=torch.tensor([[1,4,3,2],[2,3,1,5]])
            expected,other=control.prefill(tokens,capacity=4,all_logits=True);self.addCleanup(other.close)
            states=[s.new_state(2,4) for s in stages]
            for state in states:self.addCleanup(state.close)
            first=forward(stages,states,tokens[:,:2],0,True)
            torch.testing.assert_close(first,expected[:,:2],rtol=3e-5,atol=3e-6)
            for pos in (2,3):
                torch.testing.assert_close(forward(stages,states,tokens[:,pos:pos+1],pos),expected[:,pos],rtol=3e-5,atol=3e-6)

    def test_invalid_cuts_fail_before_payload_and_bad_inputs_leave_cache_usable(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path)
            with patch('damendaemon.glm_reference_loader._payload',side_effect=AssertionError('payload read')):
                for start,end in ((0,1),(1,3),(-1,2),(2,2),(False,2),(0,4)):
                    with self.subTest(start=start,end=end),self.assertRaises(ValueError):load_reference_stage(path,start,end)
        stages=prepared(model_fixture(),[(0,2),(2,3)])
        from damendaemon.glm_model import GlmModel
        self.assertTrue(all(not isinstance(stage,GlmModel) for stage in stages))
        state=stages[-1].new_state(1,4);self.addCleanup(state.close)
        for hidden in (torch.zeros(1,1,5),torch.zeros(1,1,6).half(),torch.full((1,1,6),float('nan'))):
            with self.assertRaises(ValueError):stages[-1].forward(hidden,state,position=0)
        with self.assertRaises(ValueError):stages[-1].forward(torch.zeros(1,1,6),state,position=1)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)
        self.assertEqual(stages[-1].forward(torch.zeros(1,1,6),state,position=0).shape,(1,13))

    def test_partial_failure_poisoning_and_retryable_cleanup(self):
        first,last=prepared(model_fixture(),[(0,2),(2,3)])
        state=first.new_state(1,4);self.addCleanup(state.close)
        with patch.object(first.blocks[-1],'forward',side_effect=RuntimeError('later block')):
            with self.assertRaisesRegex(RuntimeError,'later block'):first.forward(torch.tensor([[1]]),state,position=0)
        self.assertTrue(state.poisoned)
        with self.assertRaises(RuntimeError):first.forward(torch.tensor([[1]]),state,position=0)
        state=last.new_state(1,4)
        with patch.object(last,'new_state',return_value=state),patch.object(last,'_logits',side_effect=RuntimeError('head')),patch.object(state.layers[0],'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'head'):last.prefill(torch.zeros(1,1,6),capacity=4)
        self.assertEqual(last.failed_cleanups,[state]);self.assertTrue(state.closed)
        last.cleanup();self.assertFalse(last.failed_cleanups);self.assertIsNone(state.owner)

    def test_stage_boundary_and_cache_budgets_precede_work(self):
        model=model_fixture()
        first=GlmLayerStage(model.blocks[:2],start=0,total=3,embedding=model.embedding,max_output_bytes=24)
        with patch.object(first.blocks[0],'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'cache reservation'):first.new_state(2,4,max_cache_bytes=1)
        state=first.new_state(1,4);self.addCleanup(state.close)
        with patch.object(first.blocks[0],'forward',side_effect=AssertionError('executed')):
            with self.assertRaisesRegex(ValueError,'output budget'):first.forward(torch.tensor([[1,2]]),state,position=0)
            with self.assertRaisesRegex(ValueError,'final'):first.forward(torch.tensor([[1]]),state,position=0,all_logits=True)
        self.assertFalse(state.poisoned);self.assertEqual(state.position,0)
        self.assertEqual(first.forward(torch.tensor([[1]]),state,position=0).shape,(1,1,6))

    def test_wrong_stage_cache_and_partial_reorder_are_not_reusable(self):
        stages=prepared(model_fixture(),[(0,2),(2,3)])
        first=stages[0];state=first.new_state(2,4);self.addCleanup(state.close)
        with self.assertRaisesRegex(ValueError,'owner'):
            stages[-1].forward(torch.zeros(2,1,6),state,position=0)
        self.assertFalse(state.poisoned)
        first.forward(torch.tensor([[1],[2]]),state,position=0)
        with patch.object(state.layers[1],'reorder',side_effect=MemoryError('reorder')):
            with self.assertRaises(MemoryError):state.reorder(torch.tensor([1]))
        self.assertTrue(state.poisoned)
        with self.assertRaises(RuntimeError):first.forward(torch.tensor([[1]]),state,position=1)
