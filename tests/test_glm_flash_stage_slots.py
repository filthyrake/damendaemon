from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_stage_slots import FlashStageSlots
from dllm.glm_stage_slots import GlmStageSlots
from dllm.cpu_stage_slots import CpuStageSlots
from dllm.glm_flash_reference_loader import load_reference_stage
from tests.test_glm_flash_reference_loader import fixture,floating_control


class FlashStageSlotTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        config,_,values=fixture(self.path);self.control=floating_control(config,values)

    def owners(self,cuts=((0,1),(1,2),(2,3))):
        owners=[FlashStageSlots(load_reference_stage(self.path,a,b),8,3) for a,b in cuts]
        for owner in owners:self.addCleanup(owner.close)
        return owners

    def run_slots(self,owners,tokens,starts,ids,prefill):
        hidden=owners[0].embedding(tokens)
        positions=torch.stack([torch.arange(start,start+tokens.shape[1]) for start in starts])
        for owner in owners:
            hidden=owner.forward_layers(hidden,positions,torch.tensor(ids),prefill,starts[0] if prefill else 0)
            self.assertEqual(tuple(hidden.shape),(len(ids),tokens.shape[1],owner.stage.streams,owner.stage.dim))
        return owners[-1].output_logits(hidden[:,-1])

    def expected(self,tokens):
        logits,state=self.control.prefill(tokens,capacity=8,all_logits=True);state.close();return logits

    def test_packed_chunks_unequal_reordered_decode_release_reuse(self):
        prompts=[torch.tensor([[1,2,3,4]]),torch.tensor([[3,2,1,5,4]])];expected=[self.expected(p) for p in prompts]
        for cuts in (((0,1),(1,3)),((0,2),(2,3)),((0,1),(1,2),(2,3))):
            owners=self.owners(cuts)
            self.run_slots(owners,prompts[0][:,:1],[0],[2],True)
            actual=self.run_slots(owners,prompts[0][:,1:3],[1],[2],True)
            torch.testing.assert_close(actual,expected[0][:,2],rtol=3e-5,atol=3e-6)
            actual=self.run_slots(owners,prompts[1][:,:4],[0],[0],True)
            torch.testing.assert_close(actual,expected[1][:,3],rtol=3e-5,atol=3e-6)
            actual=self.run_slots(owners,torch.tensor([[4],[4]]),[4,3],[0,2],False)
            torch.testing.assert_close(actual,torch.cat([expected[1][:,4],expected[0][:,3]]),rtol=3e-5,atol=3e-6)
            self.assertTrue(all(o.lengths==[5,0,4] for o in owners))
            for owner in owners:owner.release([0])
            actual=self.run_slots(owners,prompts[0][:,:2],[0],[0],True)
            torch.testing.assert_close(actual,expected[0][:,1],rtol=3e-5,atol=3e-6)
            self.assertTrue(all(o.lengths==[2,0,4] for o in owners))

    def test_fixed_linear_plus_sparse_reservation_and_configuration_bounds(self):
        owners=self.owners();linear,sparse=owners[:2]
        self.assertEqual(linear.reserved_cache_bytes,3*linear.stage.blocks[0].attention.bytes_per_slot)
        self.assertEqual(sparse.reserved_cache_bytes,3*8*sparse.stage.blocks[0].attention.bytes_per_token)
        for owner in owners:
            with patch.object(owner.stage,'new_state',side_effect=AssertionError('allocation')):
                with self.assertRaisesRegex(ValueError,'aggregate'):FlashStageSlots(owner.stage,8,3,max_cache_bytes=owner.reserved_cache_bytes-1)
                exact=FlashStageSlots(owner.stage,8,3,max_cache_bytes=owner.reserved_cache_bytes);exact.close()
            for cls in (GlmStageSlots,CpuStageSlots):
                with self.assertRaises(ValueError):cls(owner.stage,8)
        linear.stage.config['text_config']['max_position_embeddings']=4
        with self.assertRaises(ValueError):FlashStageSlots(linear.stage,8)

    def test_invalid_geometry_positions_and_budget_leave_slots_usable(self):
        owners=self.owners();first=owners[0];hidden=first.embedding(torch.tensor([[1,2]]))
        position=torch.tensor([[0,1]]);slots=torch.tensor([0])
        for bad in (hidden.mean(2),hidden.double(),hidden[:,:,:1],torch.full_like(hidden,float('nan')),hidden[:,:0]):
            with self.assertRaises(ValueError):first.forward_layers(bad,position,slots,True)
        for positions,ids,prefill in ((torch.tensor([[0,2]]),slots,True),(position,torch.tensor([3]),True),(position,slots,False)):
            with self.assertRaises(ValueError):first.forward_layers(hidden,positions,ids,prefill)
        limit=first.stage.max_output_bytes;first.stage.max_output_bytes=hidden.numel()*4-1
        with self.assertRaises(ValueError):first.forward_layers(hidden,position,slots,True)
        first.stage.max_output_bytes=limit;self.assertEqual(first.states,[None]*3);self.assertFalse(first.failed)
        self.run_slots(owners,torch.tensor([[1,2]]),[0],[0],True)
        for ids in ([0,0],[True],[3],[1],[[]]):
            with self.assertRaises(ValueError):first.release(ids)
        self.assertEqual(first.lengths,[2,0,0]);self.assertFalse(first.failed)
        with self.assertRaises(ValueError):owners[-1].output_logits(torch.zeros(1,16))
        self.assertFalse(owners[-1].failed)

    def test_partial_row_or_allocation_failure_discards_every_slot(self):
        for allocation in (False,True):
            first=self.owners()[0];hidden=first.embedding(torch.tensor([[1],[2]]))
            method='new_state' if allocation else 'forward_activations';original=getattr(first.stage,method);calls=0
            def fail(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('second row')
                return original(*args,**kwargs)
            with patch.object(first.stage,method,fail):
                with self.assertRaisesRegex(RuntimeError,'second row'):first.forward_layers(hidden,torch.tensor([[0],[0]]),torch.tensor([0,1]),True)
            self.assertTrue(first.failed);self.assertTrue(first.closed);self.assertEqual(first.states,[None]*3)
            with self.assertRaises(RuntimeError):first.embedding(torch.tensor([[1]]))

    def test_release_failure_and_final_projection_failure_keep_retry_ownership(self):
        owners=self.owners();self.run_slots(owners,torch.tensor([[1],[2]]),[0,0],[0,1],True)
        first,last=owners[0],owners[-1];a,b=first.states[:2]
        with patch.object(a,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'release'):first.release([0])
        self.assertTrue(first.failed);self.assertTrue(b.closed);self.assertIs(first.states[0],a)
        first.close();self.assertEqual(first.states,[None]*3)
        with patch.object(last.stage,'_logits',side_effect=RuntimeError('head')):
            with self.assertRaisesRegex(RuntimeError,'head'):last.output_logits(torch.zeros(1,2,16))
        self.assertTrue(last.failed);self.assertEqual(last.states,[None]*3)
