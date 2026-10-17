from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.deepseek_stage_slots import DeepSeekStageSlots
from damendaemon.deepseek_reference_loader import load_reference_stage,load_reference_model
from damendaemon.glm_stage_slots import GlmStageSlots
from damendaemon.glm_flash_stage_slots import FlashStageSlots
from tests.test_deepseek_reference_loader import fixture
from tests.test_deepseek_checkpoint import config as geometry


class DeepSeekStageSlotTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        with patch('tests.test_deepseek_reference_loader.geometry',return_value=geometry()|dict(num_hash_layers=2)):fixture(self.path)
        self.control=load_reference_model(self.path)

    def owners(self,cuts=((0,1),(1,2),(2,3)),context=140):
        owners=[DeepSeekStageSlots(load_reference_stage(self.path,a,b),context,3) for a,b in cuts]
        for owner in owners:self.addCleanup(owner.close)
        return owners

    def run_slots(self,owners,tokens,starts,ids,prefill):
        hidden=owners[0].embedding(tokens)
        positions=torch.stack([torch.arange(start,start+tokens.shape[1]) for start in starts])
        for owner in owners:
            hidden=owner.forward_layers(hidden,positions,torch.tensor(ids),prefill,starts[0] if prefill else 0,token_ids=tokens)
            self.assertEqual(tuple(hidden.shape),(*tokens.shape,owner.stage.streams,owner.stage.dim))
        return owners[-1].output_logits(hidden[:,-1])

    def expected(self,tokens):
        logits,state=self.control.prefill(tokens,max_tokens=140);state.close();return logits

    def test_chunked_packed_hash_routes_compressed_boundaries_unequal_decode_release_reuse(self):
        for cuts,length in ((((0,1),(1,3)),8),(((0,2),(2,3)),129),(((0,1),(1,2),(2,3)),9)):
            owners=self.owners(cuts);tokens=torch.arange(length+1)[None]%16
            self.run_slots(owners,tokens[:,:3],[0],[2],True)
            actual=self.run_slots(owners,tokens[:,3:length],[3],[2],True)
            torch.testing.assert_close(actual,self.expected(tokens[:,:length]),rtol=4e-5,atol=4e-6)
            other=torch.tensor([[4,3,2,1,0]])
            self.run_slots(owners,other[:,:4],[0],[0],True)
            actual=self.run_slots(owners,torch.tensor([[0],[int(tokens[0,length])]]),[4,length],[0,2],False)
            torch.testing.assert_close(actual,torch.cat([self.expected(other),self.expected(tokens)]),rtol=4e-5,atol=4e-6)
            self.assertTrue(all(o.lengths==[5,0,length+1] for o in owners))
            for owner in owners:owner.release([0])
            actual=self.run_slots(owners,other[:,:2],[0],[0],True)
            torch.testing.assert_close(actual,self.expected(other[:,:2]),rtol=4e-5,atol=4e-6)

    def test_exact_aggregate_reservation_and_family_controls_before_allocation(self):
        for owner in self.owners():
            reserved=3*owner.stage.cache_bytes_per_slot(140);self.assertEqual(reserved,owner.reserved_cache_bytes)
            with patch.object(owner.stage,'prefill_activations',side_effect=AssertionError('allocate')):
                with self.assertRaisesRegex(ValueError,'aggregate'):DeepSeekStageSlots(owner.stage,140,3,max_cache_bytes=reserved-1)
                exact=DeepSeekStageSlots(owner.stage,140,3,max_cache_bytes=reserved);exact.close()
            for cls in (GlmStageSlots,FlashStageSlots):
                with self.assertRaises(ValueError):cls(owner.stage,140)
        owner.stage.config['max_position_embeddings']=4
        with self.assertRaises(ValueError):DeepSeekStageSlots(owner.stage,8)

    def test_invalid_tokens_streams_positions_and_limits_leave_slots_usable(self):
        owners=self.owners();first=owners[0];tokens=torch.tensor([[1,2]]);hidden=first.embedding(tokens);positions=torch.tensor([[0,1]]);slots=torch.tensor([0])
        with patch.object(first.stage,'prefill_activations',side_effect=AssertionError('allocate')):
            for bad in (None,tokens.double(),tokens[:,:1],torch.tensor([[16,1]]),torch.tensor([[-1,1]])):
                with self.assertRaises(ValueError):first.forward_layers(hidden,positions,slots,True,token_ids=bad)
            for bad in (hidden.mean(2),hidden[:,:0],hidden[:0],torch.full_like(hidden,float('nan'))):
                with self.assertRaises(ValueError):first.forward_layers(bad,positions,slots,True,token_ids=tokens)
            for pos,ids in ((torch.tensor([[0,2]]),slots),(positions,torch.tensor([3]))):
                with self.assertRaises(ValueError):first.forward_layers(hidden,pos,ids,True,token_ids=tokens)
            with self.assertRaises(ValueError):first.forward_layers(hidden.expand(2,-1,-1,-1),positions.expand(2,-1),torch.tensor([0,0]),True,token_ids=tokens.expand(2,-1))
        self.assertEqual(first.states,[None]*3);self.assertFalse(first.failed)
        self.run_slots(owners,tokens,[0],[0],True)
        with self.assertRaises(ValueError):first.forward_layers(hidden,positions,slots,True,token_ids=tokens)
        for ids in ([0,0],[True],[1],[3]):
            with self.assertRaises(ValueError):first.release(ids)
        self.assertEqual(first.lengths,[2,0,0]);self.assertFalse(first.failed)

    def test_partial_row_prefill_or_decode_failure_retires_all_slots(self):
        for prefill in (True,False):
            owners=self.owners();first=owners[0]
            if not prefill:self.run_slots(owners,torch.tensor([[1],[2]]),[0,0],[0,1],True)
            tokens=torch.tensor([[2],[3]]);hidden=first.embedding(tokens);method='prefill_activations' if prefill else 'decode_activations';original=getattr(first.stage,method);calls=0
            def fail(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('second row')
                return original(*args,**kwargs)
            with patch.object(first.stage,method,fail):
                with self.assertRaisesRegex(RuntimeError,'second row'):first.forward_layers(hidden,torch.tensor([[0 if prefill else 1]]*2),torch.tensor([0,1]),prefill,token_ids=tokens)
            self.assertTrue(first.failed);self.assertTrue(first.closed);self.assertEqual(first.states,[None]*3)

    def test_projection_release_and_failed_prefill_cleanup_keep_retry_owner(self):
        from damendaemon.deepseek_layer_stage import DeepSeekStageState
        owners=self.owners();self.run_slots(owners,torch.tensor([[1],[2]]),[0,0],[0,1],True)
        first,last=owners[0],owners[-1];a,b=first.states[:2]
        with patch.object(a,'close',side_effect=RuntimeError('release')):
            with self.assertRaisesRegex(RuntimeError,'release'):first.release([0])
        self.assertTrue(first.failed);self.assertTrue(b.closed);self.assertIs(first.states[0],a);first.close()
        with patch.object(last.stage,'_logits',side_effect=RuntimeError('head')):
            with self.assertRaisesRegex(RuntimeError,'head'):last.output_logits(torch.zeros(1,2,32))
        self.assertTrue(last.failed);self.assertEqual(last.states,[None]*3)
        owner=self.owners(((0,2),(2,3)))[0];tokens=torch.tensor([[1]])
        with patch.object(owner.stage.blocks[-1],'prefill',side_effect=MemoryError('later')),patch.object(DeepSeekStageState,'close',side_effect=OSError('cleanup')):
            with self.assertRaises(MemoryError):owner.forward_layers(owner.embedding(tokens),torch.tensor([[0]]),torch.tensor([0]),True,token_ids=tokens)
            self.assertTrue(owner.failed);self.assertTrue(owner.stage.prefill_cleanup_pending)
            with self.assertRaises(OSError):owner.close()
        owner.close();self.assertFalse(owner.stage.prefill_cleanup_pending)
