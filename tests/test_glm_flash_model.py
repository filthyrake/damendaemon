import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_model import FlashModel
from tests import test_glm_flash_block


def fixture():
    blocks=[test_glm_flash_block.fixture(sparse)[0] for sparse in (False,True,False)]
    torch.manual_seed(1041)
    return FlashModel(torch.randn(13,4),blocks,torch.randn(4),torch.randn(13,4)*.2)


def control(tokens,c):
    hidden=c.embedding[tokens].unsqueeze(-2).expand(-1,-1,2,-1)
    for sparse,block in zip((False,True,False),c.blocks):hidden=test_glm_flash_block.control(hidden,block,sparse)
    value=(hidden[:,:,0]+hidden[:,:,1])/2
    return (value/torch.sqrt(value.square().mean(-1,keepdim=True)+c.epsilon)*c.norm)@c.head.T


class FlashModelTests(unittest.TestCase):
    def setUp(self):
        self.model=fixture();self.tokens=torch.tensor([[1,3,2,5,4],[2,4,3,6,5]])

    def test_independent_mixed_stack_chunk_decode_logits(self):
        c=self.model;expected=control(self.tokens,c)
        actual,state=c.prefill(self.tokens,capacity=5,all_logits=True);self.addCleanup(state.close)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=6e-5)
        prefix,state=c.prefill(self.tokens[:,:2],capacity=5,all_logits=True);self.addCleanup(state.close)
        torch.testing.assert_close(prefix,expected[:,:2],atol=2e-6,rtol=6e-5)
        for position in range(2,5):
            logits=c.forward(self.tokens[:,position:position+1],state,position=position)
            torch.testing.assert_close(logits,expected[:,position],atol=2e-6,rtol=6e-5)
        self.assertEqual(state.position,5)

    def test_mixed_model_packed_attention_and_routed_shared_ffns(self):
        from tests import test_glm_flash_ffn,test_glm_flash_sparse
        from tests.test_glm_block import ffn_weights
        from damendaemon.glm_flash_ffn import FlashFeedForward
        from damendaemon.glm_flash_sparse import FlashSparseAttention
        packed=fixture();floating=fixture()
        attention,decoded,_=test_glm_flash_sparse.fixture(True)
        packed.blocks[1].attention=attention
        floating.blocks[1].attention=FlashSparseAttention(decoded,attention.indexer,heads=2,head_dim=3,value_dim=2)
        for i in range(3):
            sparse=i>0
            if not sparse:weights=packed.blocks[i].feed_forward.weights
            else:
                weights=ffn_weights(True)
                for name,value in list(weights.items()):
                    if name.endswith('down_proj.weight'):weights[name]=value[...,:4,:]
                    elif not name.endswith('correction_bias'):weights[name]=value[...,:4]
            prepared,decoded=test_glm_flash_ffn.packed_ffn(weights,sparse,.5)
            packed.blocks[i].feed_forward=prepared
            floating.blocks[i].feed_forward=FlashFeedForward(decoded,kind='sparse' if sparse else 'dense',top_k=2,activation_limit=.5)
        expected,control_state=floating.prefill(self.tokens,capacity=5,all_logits=True);self.addCleanup(control_state.close)
        state=packed.new_state(2,5);self.addCleanup(state.close)
        for position in range(5):
            actual=packed.forward(self.tokens[:,position:position+1],state,position=position)
            torch.testing.assert_close(actual,expected[:,position],atol=3e-6,rtol=8e-5)

    def test_aggregate_cache_reservation_reorder_shrink(self):
        c=self.model;per_slot=sum(block.attention.bytes_per_slot for block in (c.blocks[0],c.blocks[2]))+5*c.blocks[1].attention.bytes_per_token
        self.assertEqual(c._cache_per_slot(5),per_slot)
        with patch.object(c.blocks[0],'new_state',side_effect=AssertionError('allocated before admission')):
            with self.assertRaises(ValueError):c.new_state(2,5,max_cache_bytes=2*per_slot-1)
        _,state=c.prefill(self.tokens[:,:2],capacity=5,max_cache_bytes=3*per_slot);self.addCleanup(state.close)
        order=torch.tensor([1,0,1]);state.reorder(order)
        logits=c.forward(self.tokens[order,2:3],state,position=2)
        torch.testing.assert_close(logits,control(self.tokens[order,:3],c)[:,-1],atol=2e-6,rtol=6e-5)
        state.reorder(torch.tensor([2]));self.assertEqual(state.layers[0].attention.recurrent.shape[0],1)
        with self.assertRaises(ValueError):state.reorder(torch.tensor([0,0,0,0]))

    def test_head_and_partial_reorder_failure_poison_whole_model(self):
        c=self.model;state=c.new_state(2,5);self.addCleanup(state.close)
        with patch.object(c,'_logits',side_effect=RuntimeError('head')):
            with self.assertRaisesRegex(RuntimeError,'head'):c.forward(self.tokens[:,:1],state,position=0)
        self.assertTrue(state.poisoned);self.assertTrue(all(layer.poisoned for layer in state.layers));self.assertEqual(state.position,0)
        with self.assertRaises(RuntimeError):c.forward(self.tokens[:,:1],state,position=0)
        _,state=c.prefill(self.tokens[:,:1],capacity=5);self.addCleanup(state.close)
        with patch.object(state.layers[1],'reorder',side_effect=MemoryError('second layer')):
            with self.assertRaises(MemoryError):state.reorder(torch.tensor([1]))
        self.assertTrue(state.poisoned)

    def test_cleanup_retained_after_failed_prefill_and_allocation(self):
        c=self.model;state=c.new_state(2,5)
        with patch.object(c,'new_state',return_value=state),patch.object(c,'_logits',side_effect=RuntimeError('primary')),patch.object(state.layers[0],'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary'):c.prefill(self.tokens[:,:1],capacity=5)
        self.assertEqual(c.failed_cleanups,[state]);self.assertEqual(len(state.layers),1)
        c.cleanup();self.assertFalse(c.failed_cleanups);self.assertIsNone(state.owner)
        created=[];original=c.blocks[0].new_state
        def allocate(*args,**kwargs):
            state=original(*args,**kwargs);created.append(state);return state
        with patch.object(c.blocks[0],'new_state',side_effect=allocate),patch.object(c.blocks[1],'new_state',side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):c.new_state(2,5)
        self.assertTrue(created[0].attention.closed)

    def test_invalid_tokens_owners_cursors_and_output_budget(self):
        c=self.model;state=c.new_state(2,5);self.addCleanup(state.close)
        for tokens in (self.tokens.float(),self.tokens+13,self.tokens[:,:0]):
            with self.assertRaises(ValueError):c.forward(tokens,state,position=0)
        with self.assertRaises(ValueError):fixture().forward(self.tokens,state,position=0)
        c.max_output_bytes=1
        with patch.object(c.blocks[0],'forward',side_effect=AssertionError('executed')):
            with self.assertRaises(ValueError):c.forward(self.tokens,state,position=0)
        c.max_output_bytes=64*1024**2;state.layers[1].attention.position=1
        with patch.object(c.blocks[0],'forward',side_effect=AssertionError('executed')):
            with self.assertRaisesRegex(RuntimeError,'cursors'):c.forward(self.tokens,state,position=0)
        self.assertTrue(state.poisoned)
