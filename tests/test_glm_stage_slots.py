from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_stage_slots import GlmStageSlots
from dllm.glm_reference_loader import load_reference_stage
from tests.test_glm_reference_loader import fixture, floating_control
from tests.test_glm_model import model_fixture
from tests.test_glm_layer_stage import prepared


class GlmStageSlotTests(unittest.TestCase):
    def owners(self):
        stages=prepared(model_fixture(),[(0,2),(2,3)])
        owners=[GlmStageSlots(stage,8,3) for stage in stages]
        for owner in owners:self.addCleanup(owner.close)
        return owners

    def run_slots(self, owners, tokens, starts, ids, prefill):
        hidden=owners[0].embedding(tokens)
        positions=torch.stack([torch.arange(start,start+tokens.shape[1]) for start in starts])
        for owner in owners:
            hidden=owner.forward_layers(hidden,positions,torch.tensor(ids),prefill,starts[0] if prefill else 0)
        return owners[-1].output_logits(hidden[:,-1])

    def test_packed_unequal_cursors_reordered_decode_and_release_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,_,values=fixture(path)
            control=floating_control(config,values)
            owners=[GlmStageSlots(load_reference_stage(path,start,end),8,3) for start,end in ((0,2),(2,3))]
            for owner in owners:self.addCleanup(owner.close)
            prompts=[torch.tensor([[1,2,3,4]]),torch.tensor([[3,2,1,5,4]])]
            expected=[]
            for prompt in prompts:
                logits,state=control.prefill(prompt,capacity=8,all_logits=True);state.close();expected.append(logits)
            self.run_slots(owners,prompts[0][:,:1],[0],[2],True)
            actual=self.run_slots(owners,prompts[0][:,1:3],[1],[2],True)
            torch.testing.assert_close(actual,expected[0][:,2],rtol=3e-5,atol=3e-6)
            actual=self.run_slots(owners,prompts[1][:,:4],[0],[0],True)
            torch.testing.assert_close(actual,expected[1][:,3],rtol=3e-5,atol=3e-6)
            actual=self.run_slots(owners,torch.tensor([[4],[4]]),[4,3],[0,2],False)
            torch.testing.assert_close(actual,torch.cat([expected[1][:,4],expected[0][:,3]]),rtol=3e-5,atol=3e-6)
            self.assertTrue(all(owner.lengths==[5,0,4] for owner in owners))
            for owner in owners:owner.release([0])
            actual=self.run_slots(owners,prompts[0][:,:2],[0],[0],True)
            torch.testing.assert_close(actual,expected[0][:,1],rtol=3e-5,atol=3e-6)
            self.assertTrue(all(owner.lengths==[2,0,4] for owner in owners))

    def test_aggregate_reservation_and_preflight_leave_slots_usable(self):
        owners=self.owners();first=owners[0]
        with patch.object(first.stage,'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'aggregate'):
                GlmStageSlots(first.stage,8,3,max_cache_bytes=first.reserved_cache_bytes-1)
        hidden=first.embedding(torch.tensor([[1,2]]))
        for positions,ids,prefill in ((torch.tensor([[0,2]]),torch.tensor([0]),True),
                                    (torch.tensor([[0,1]]),torch.tensor([3]),True),
                                    (torch.tensor([[0,1]]),torch.tensor([0]),False)):
            with self.assertRaises(ValueError):first.forward_layers(hidden,positions,ids,prefill)
        self.assertEqual(first.states,[None]*3);self.assertFalse(first.failed)
        self.run_slots(owners,torch.tensor([[1,2]]),[0],[0],True)
        for slots in ([0,0],[True],[3],[1],[[]]):
            with self.assertRaises(ValueError):first.release(slots)
        self.assertFalse(first.failed);self.assertEqual(first.lengths,[2,0,0])
        with self.assertRaises(ValueError):first.forward_layers(hidden,torch.tensor([[0,1]]),torch.tensor([0]),True)
        self.assertFalse(first.failed)

    def test_partial_row_failure_closes_every_slot_and_cannot_resume(self):
        first,last=self.owners();hidden=first.embedding(torch.tensor([[1],[2]]))
        original=first.stage.forward_activations;calls=0
        def fail(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('second row')
            return original(*args,**kwargs)
        with patch.object(first.stage,'forward_activations',fail):
            with self.assertRaisesRegex(RuntimeError,'second row'):
                first.forward_layers(hidden,torch.tensor([[0],[0]]),torch.tensor([0,1]),True)
        self.assertTrue(first.failed);self.assertTrue(first.closed);self.assertEqual(first.states,[None]*3)
        with self.assertRaises(RuntimeError):first.embedding(torch.tensor([[1]]))

    def test_release_failure_attempts_other_slots_and_preserves_retry_owner(self):
        owners=self.owners();self.run_slots(owners,torch.tensor([[1],[2]]),[0,0],[0,1],True)
        first=owners[0];a,b=first.states[:2]
        with patch.object(a,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'release'):first.release([0])
        self.assertTrue(first.failed);self.assertTrue(first.closed);self.assertTrue(b.closed)
        self.assertIs(first.states[0],a);self.assertIsNone(first.states[1])
        first.close();self.assertEqual(first.states,[None]*3)

    def test_head_failure_and_hidden_boundary_validation(self):
        owners=self.owners();first,last=owners
        with self.assertRaises(ValueError):last.embedding(torch.tensor([[1]]))
        state=first.stage.new_state(1,8);self.addCleanup(state.close)
        for hidden in (torch.zeros(1,1,6).half(),torch.full((1,1,6),float('nan')),torch.zeros(1,1,5)):
            with self.assertRaises(ValueError):first.stage.forward_activations(hidden,state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)
        self.run_slots(owners,torch.tensor([[1]]),[0],[0],True)
        with patch.object(last.stage,'_logits',side_effect=RuntimeError('head')):
            with self.assertRaisesRegex(RuntimeError,'head'):last.output_logits(torch.zeros(1,6))
        self.assertTrue(last.failed);self.assertEqual(last.states,[None]*3)
