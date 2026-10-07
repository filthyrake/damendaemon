import unittest
from unittest.mock import patch
import torch
from dllm.glm_model import GlmModel
from dllm.glm_block import GlmBlock
from dllm.glm_ffn import GlmFeedForward
from tests.test_glm_cached_attention import fixture
from tests.test_glm_block import ffn_weights


def model_fixture():
    blocks=[]
    for index,shared in enumerate((False,True,False)):
        attention,_=fixture(shared)
        ff=GlmFeedForward(ffn_weights(index>0),kind='sparse' if index>0 else 'dense',top_k=2)
        blocks.append(GlmBlock(attention,ff,torch.ones(6)*(1+index*.1),torch.ones(6)))
    generator=torch.Generator().manual_seed(81)
    return GlmModel(torch.randn(13,6,generator=generator),blocks,torch.ones(6),torch.randn(13,6,generator=generator))


class GlmModelTests(unittest.TestCase):
    def setUp(self):
        self.model=model_fixture()
        self.tokens=torch.tensor([[1,3,2,5,4],[2,4,3,6,5]])

    def test_incremental_logits_match_full_prefix_and_manual_stack(self):
        model=self.model
        whole,state=model.prefill(self.tokens,capacity=5,all_logits=True)
        self.addCleanup(state.close)
        prefix,cache=model.prefill(self.tokens[:,:2],capacity=5,all_logits=True)
        self.addCleanup(cache.close)
        torch.testing.assert_close(prefix,whole[:,:2],rtol=2e-5,atol=2e-6)
        for position in range(2,5):
            result=model.forward(self.tokens[:,position:position+1],cache,position=position)
            torch.testing.assert_close(result,whole[:,position],rtol=2e-5,atol=2e-6)
            self.assertEqual(cache.position,position+1)
        hidden=model.embedding[self.tokens];selected=None
        for block in model.blocks:
            hidden,selected,layer=block.prefill(hidden,capacity=5,previous_indices=selected if block.attention.indexer is None else None)
            layer.close()
        expected=(hidden/(hidden.square().mean(-1,keepdim=True)+1e-5).sqrt()*model.norm)@model.head.T
        torch.testing.assert_close(whole,expected,rtol=2e-5,atol=2e-6)

    def test_global_budget_before_layer_allocation_and_reordered_slots(self):
        model=self.model
        with patch.object(model.blocks[0],'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError):model.new_state(2,5,max_cache_bytes=1)
        _,state=model.prefill(self.tokens[:,:2],capacity=5);self.addCleanup(state.close)
        order=torch.tensor([1,0,1]);state.reorder(order)
        actual=model.forward(self.tokens[order,2:3],state,position=2)
        expected,fresh=model.prefill(self.tokens[order,:3],capacity=5)
        try:torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        finally:fresh.close()
        state.reorder(torch.tensor([2]));self.assertEqual(state.layers[0].attention.latents.shape[0],1)

    def test_head_and_partial_reorder_failures_poison_every_layer(self):
        model=self.model;state=model.new_state(2,5);self.addCleanup(state.close)
        with patch.object(model,'_logits',side_effect=RuntimeError('head failure')):
            with self.assertRaises(RuntimeError):model.forward(self.tokens[:,:1],state,position=0)
        self.assertTrue(state.poisoned);self.assertEqual(state.position,0)
        self.assertTrue(all(layer.poisoned and layer.attention.poisoned for layer in state.layers))
        with self.assertRaises(RuntimeError):model.forward(self.tokens[:,:1],state,position=0)
        _,other=model.prefill(self.tokens[:,:1],capacity=5);self.addCleanup(other.close)
        with patch.object(other.layers[1],'reorder',side_effect=MemoryError('second layer')):
            with self.assertRaises(MemoryError):other.reorder(torch.tensor([1]))
        self.assertTrue(other.poisoned)

    def test_failed_prefill_cleanup_is_retained_and_retried(self):
        model=self.model;state=model.new_state(2,5)
        with patch.object(model,'new_state',return_value=state),patch.object(model,'_logits',side_effect=RuntimeError('primary')), \
                patch.object(state.layers[0],'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary'):model.prefill(self.tokens[:,:1],capacity=5)
        self.assertEqual(model.failed_cleanups,[state]);self.assertTrue(state.closed)
        self.assertEqual(len(state.layers),1)
        model.cleanup()
        self.assertFalse(model.failed_cleanups);self.assertFalse(state.layers);self.assertIsNone(state.owner)

    def test_bad_tokens_and_owner_do_not_advance_caches(self):
        model=self.model;state=model.new_state(2,5);self.addCleanup(state.close)
        for tokens in (self.tokens.float(),self.tokens+13,self.tokens[:,:0]):
            with self.assertRaises(ValueError):model.forward(tokens,state,position=0)
        other=model_fixture()
        with self.assertRaises(ValueError):other.forward(self.tokens[:,:1],state,position=0)
        self.assertFalse(state.poisoned);self.assertEqual(state.position,0)
        with self.assertRaises(ValueError):model.forward(self.tokens[:,:1],state,position=1)

    def test_layer_allocation_failure_releases_earlier_states(self):
        created=[]
        original=self.model.blocks[0].new_state
        def allocate(*args,**kwargs):
            state=original(*args,**kwargs);created.append(state);return state
        with patch.object(self.model.blocks[0],'new_state',side_effect=allocate), \
                patch.object(self.model.blocks[1],'new_state',side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):self.model.new_state(2,5)
        self.assertTrue(created[0].attention.closed)
        self.assertIsNone(created[0].attention.latents)
        self.assertFalse(self.model.failed_cleanups)

    def test_layer_cursor_drift_is_detected_before_more_execution(self):
        state=self.model.new_state(2,5);self.addCleanup(state.close)
        state.layers[1].attention.position=1
        with patch.object(self.model.blocks[0],'forward',side_effect=AssertionError('executed')):
            with self.assertRaises(RuntimeError):self.model.forward(self.tokens[:,:1],state,position=0)
        self.assertTrue(state.poisoned)
