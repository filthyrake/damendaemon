import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_ffn import GlmFeedForward,select_glm_experts
from damendaemon.glm_block import GlmBlock
from tests.test_glm_cached_attention import fixture


def ffn_weights(sparse=True):
    generator=torch.Generator().manual_seed(27)
    def tensor(*shape):return torch.randn(*shape,generator=generator)*.2
    if not sparse:return {'gate_proj.weight':tensor(9,6),'up_proj.weight':tensor(9,6),'down_proj.weight':tensor(6,9)}
    return {'gate.weight':tensor(3,6),'gate.e_score_correction_bias':tensor(3),
            'experts.gate_proj.weight':tensor(3,8,6),'experts.up_proj.weight':tensor(3,8,6),'experts.down_proj.weight':tensor(3,6,8),
            'shared_experts.gate_proj.weight':tensor(8,6),'shared_experts.up_proj.weight':tensor(8,6),'shared_experts.down_proj.weight':tensor(6,8)}


def dense(x,gate,up,down):
    g=x@gate.T
    return (g.sigmoid()*g*(x@up.T))@down.T


class GlmBlockTests(unittest.TestCase):
    def test_routing_correction_and_additive_underflow_denominator(self):
        logits=torch.tensor([[-46.,-47.,-48.],[1.,2.,3.]])
        bias=torch.tensor([1.,0.,-.1])
        weights,ids=select_glm_experts(logits,bias,2)
        expected_ids=(logits.sigmoid()+bias).argsort(dim=-1,descending=True,stable=True)[:,:2]
        probabilities=logits.sigmoid().gather(-1,expected_ids)
        expected=probabilities/(probabilities.sum(-1,keepdim=True)+1e-20)*2.5
        torch.testing.assert_close(ids,expected_ids,rtol=0,atol=0)
        torch.testing.assert_close(weights,expected,rtol=0,atol=0)
        self.assertLess(float(weights[0].sum()),2.5)

    def test_dense_and_sparse_ffn_match_independent_expert_equations(self):
        _,hidden=fixture()
        for sparse in (False,True):
            w=ffn_weights(sparse)
            ffn=GlmFeedForward(w,kind='sparse' if sparse else 'dense',top_k=2,workspace_bytes=8192)
            if not sparse:expected=dense(hidden,w['gate_proj.weight'],w['up_proj.weight'],w['down_proj.weight'])
            else:
                flat=hidden.reshape(-1,6)
                probabilities=(flat@w['gate.weight'].T).sigmoid()
                ids=(probabilities+w['gate.e_score_correction_bias']).argsort(dim=-1,descending=True,stable=True)[:,:2]
                coefficients=probabilities.gather(-1,ids);coefficients=coefficients/(coefficients.sum(-1,keepdim=True)+1e-20)*2.5
                expected=dense(flat,w['shared_experts.gate_proj.weight'],w['shared_experts.up_proj.weight'],w['shared_experts.down_proj.weight'])
                for row in range(flat.shape[0]):
                    for slot in range(2):
                        expert=int(ids[row,slot])
                        expected[row]+=dense(flat[row],w['experts.gate_proj.weight'][expert],w['experts.up_proj.weight'][expert],w['experts.down_proj.weight'][expert])*coefficients[row,slot]
                expected=expected.reshape_as(hidden)
            torch.testing.assert_close(ffn(hidden),expected,rtol=2e-5,atol=2e-6)

    def test_block_cached_output_matches_fresh_prefix(self):
        for sparse in (False,True):
            attention,hidden=fixture()
            ffn=GlmFeedForward(ffn_weights(sparse),kind='sparse' if sparse else 'dense',top_k=2)
            block=GlmBlock(attention,ffn,torch.ones(6),torch.ones(6))
            state=block.new_state(2,7);self.addCleanup(state.close)
            for position in range(7):
                output,_=block.forward(hidden[:,position:position+1],state,position=position)
                expected,_,fresh=block.prefill(hidden[:,:position+1],capacity=7)
                try:torch.testing.assert_close(output,expected[:,-1:],rtol=2e-5,atol=2e-6)
                finally:fresh.close()
                self.assertEqual(state.position,position+1)

    def test_normalization_failure_preserves_or_poisons_state_at_the_right_boundary(self):
        attention,hidden=fixture()
        block=GlmBlock(attention,GlmFeedForward(ffn_weights(),top_k=2),torch.ones(6),torch.ones(6))
        state=block.new_state(2,7);self.addCleanup(state.close)
        with patch.object(attention,'forward',wraps=attention.forward) as forward:
            with self.assertRaisesRegex(RuntimeError,'normalization variance'):
                block.forward(torch.full_like(hidden[:,:1],1e20),state,position=0)
            forward.assert_not_called()
        self.assertFalse(state.poisoned);self.assertEqual(state.attention.position,0)
        original=attention.forward
        def oversized(*args,**kwargs):
            attended,indices=original(*args,**kwargs)
            return torch.full_like(attended,1e20),indices
        with patch.object(attention,'forward',side_effect=oversized):
            with self.assertRaisesRegex(RuntimeError,'normalization variance'):
                block.forward(hidden[:,:1],state,position=0)
        self.assertTrue(state.poisoned);self.assertTrue(state.attention.poisoned)
        self.assertEqual(state.position,0);self.assertEqual(state.attention.position,1)

    def test_ffn_failure_after_attention_poisoned_without_block_cursor_commit(self):
        attention,hidden=fixture();ffn=GlmFeedForward(ffn_weights(),top_k=2)
        block=GlmBlock(attention,ffn,torch.ones(6),torch.ones(6));state=block.new_state(2,7)
        self.addCleanup(state.close)
        with patch.object(GlmFeedForward,'__call__',side_effect=RuntimeError('FFN failure')):
            with self.assertRaises(RuntimeError):block.forward(hidden[:,:1],state,position=0)
        self.assertTrue(state.poisoned);self.assertTrue(state.attention.poisoned)
        self.assertEqual(state.position,0);self.assertEqual(state.attention.position,1)
        with self.assertRaises(RuntimeError):block.forward(hidden[:,:1],state,position=0)
        with self.assertRaises(ValueError):GlmFeedForward(ffn_weights(),top_k=2,workspace_bytes=1)
