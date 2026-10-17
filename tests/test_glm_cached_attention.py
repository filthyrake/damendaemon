import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_cached_attention import CachedGlmAttention
from tests import test_glm_attention,test_glm_indexer


def fixture(shared=False):
    attention=test_glm_attention.GlmAttentionTests();attention.setUp()
    indexer=test_glm_indexer.GlmIndexerTests();indexer.setUp()
    weights=attention.weights|{'kv_b_proj.weight':attention.kv,'o_proj.weight':torch.arange(36,dtype=torch.float32).reshape(6,6)/100}
    if not shared:weights|={'indexer.'+name:value for name,value in indexer.weights.items()}
    return CachedGlmAttention(weights,heads=2,nope_dim=2,rope_dim=4,value_dim=3,index_top_k=3,
                              indexer_type='shared' if shared else 'full'),attention.hidden


class CachedGlmAttentionTests(unittest.TestCase):
    def test_chunk_prefill_and_decode_match_fresh_prefix(self):
        model,hidden=fixture()
        state=model.new_state(2,7)
        self.addCleanup(state.close)
        for start,stop in ((0,2),(2,5),(5,6),(6,7)):
            actual,selected=model.forward(hidden[:,start:stop],state,position=start)
            expected,ids,fresh=model.prefill(hidden[:,:stop],capacity=7)
            try:
                torch.testing.assert_close(actual,expected[:,start:stop],rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(selected,ids[:,start:stop],rtol=0,atol=0)
            finally:fresh.close()
            self.assertEqual(state.position,stop)
        with self.assertRaises(ValueError):model.forward(hidden[:,:1],state,position=7)
        self.assertFalse(state.poisoned)

    def test_shared_layer_reuses_full_layer_selection(self):
        full,hidden=fixture();shared,_=fixture(True)
        a=full.new_state(2,7);b=shared.new_state(2,7)
        self.addCleanup(a.close);self.addCleanup(b.close)
        for position in range(7):
            _,ids=full.forward(hidden[:,position:position+1],a,position=position)
            output,reused=shared.forward(hidden[:,position:position+1],b,position=position,previous_indices=ids)
            _,fresh_ids,fresh_full=full.prefill(hidden[:,:position+1],capacity=7)
            expected,_,fresh_shared=shared.prefill(hidden[:,:position+1],capacity=7,previous_indices=fresh_ids)
            try:torch.testing.assert_close(output,expected[:,-1:],rtol=2e-5,atol=2e-6)
            finally:fresh_full.close();fresh_shared.close()
            self.assertIs(reused,ids)

    def test_reorder_shrink_duplicate_slots_and_failure_atomicity(self):
        model,hidden=fixture();_,_,state=model.prefill(hidden[:,:3],capacity=7)
        self.addCleanup(state.close)
        before=state.latents
        original=torch.Tensor.index_select
        calls=0
        def allocate(tensor,*args):
            nonlocal calls
            calls+=1
            if calls==2:raise MemoryError('second allocation')
            return original(tensor,*args)
        with patch.object(torch.Tensor,'index_select',new=allocate):
            with self.assertRaises(MemoryError):state.reorder(torch.tensor([1,0]))
        self.assertEqual(calls,2)
        self.assertIs(state.latents,before);self.assertFalse(state.poisoned)
        order=torch.tensor([1,0,1]);state.reorder(order)
        actual,_=model.forward(hidden[order,3:4],state,position=3)
        expected,_,fresh=model.prefill(hidden[order,:4],capacity=7)
        try:torch.testing.assert_close(actual,expected[:,-1:],rtol=2e-5,atol=2e-6)
        finally:fresh.close()
        state.reorder(torch.tensor([2]))
        self.assertEqual(state.latents.shape[0],1)
        self.assertEqual(state.position,4)

    def test_owner_budget_and_partial_mutation_failure(self):
        model,hidden=fixture();other,_=fixture()
        with patch('damendaemon.glm_cached_attention.torch.empty',side_effect=AssertionError('cache allocated')):
            with self.assertRaises(ValueError):model.new_state(2,7,max_cache_bytes=1)
        state=model.new_state(2,7);self.addCleanup(state.close)
        with self.assertRaises(ValueError):other.forward(hidden[:,:1],state,position=0)
        with self.assertRaises(ValueError):model.forward(hidden[:,:1],state,position=1)
        self.assertFalse(state.poisoned)
        with patch('damendaemon.glm_cached_attention.sparse_latent_attention',side_effect=RuntimeError('after cache write')):
            with self.assertRaises(RuntimeError):model.forward(hidden[:,:1],state,position=0)
        self.assertTrue(state.poisoned);self.assertEqual(state.position,0)
        with self.assertRaises(RuntimeError):model.forward(hidden[:,:1],state,position=0)
        state.close();state.close()
        self.assertIsNone(state.latents);self.assertIsNone(state.owner)

    def test_prefill_failure_releases_owned_state_and_reorder_budget(self):
        model,hidden=fixture();created=[]
        allocate=model.new_state
        def capture(*args,**kwargs):
            state=allocate(*args,**kwargs);created.append(state);return state
        with patch.object(model,'new_state',side_effect=capture),patch('damendaemon.glm_cached_attention.sparse_latent_attention',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):model.prefill(hidden[:,:1],capacity=7)
        self.assertTrue(created[0].closed);self.assertIsNone(created[0].latents)
        state=model.new_state(1,7);self.addCleanup(state.close)
        state.max_cache_bytes=state.bytes_per_slot
        with patch.object(torch.Tensor,'index_select',side_effect=AssertionError('allocation')):
            with self.assertRaises(ValueError):state.reorder(torch.tensor([0,0]))
        self.assertFalse(state.poisoned)

    def test_missing_shared_selection_rejected_before_mutation(self):
        model,hidden=fixture(True)
        state=model.new_state(2,7);self.addCleanup(state.close)
        with self.assertRaises(ValueError):model.forward(hidden[:,:1],state,position=0)
        self.assertFalse(state.poisoned);self.assertEqual(state.position,0)
