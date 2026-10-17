import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_block import FlashBlock
from damendaemon.glm_flash_hyper import FlashHyperConnection
from damendaemon.glm_flash_ffn import FlashFeedForward
from tests import test_glm_flash_linear,test_glm_flash_sparse,test_glm_flash_hyper


def fixture(sparse):
    attention=test_glm_flash_sparse.fixture()[0] if sparse else test_glm_flash_linear.layer()
    torch.manual_seed(1001)
    def hyper():return FlashHyperConnection(torch.randn(8,8)*.2,torch.randn(8)*.1,torch.tensor([.2,.3,.4]),streams=2)
    weights={'gate_proj.weight':torch.randn(7,4)*.3,'up_proj.weight':torch.randn(7,4)*.3,'down_proj.weight':torch.randn(4,7)*.3}
    block=FlashBlock(attention,FlashFeedForward(weights,kind='dense',activation_limit=.5),hyper(),hyper(),torch.randn(4),torch.randn(4))
    return block,torch.randn(2,5,2,4)


def control(hidden,c,sparse):
    def norm(x,w):return x/torch.sqrt(x.square().mean(-1,keepdim=True)+c.epsilon)*w
    def restore(x,residual,post,matrix):
        result=torch.zeros_like(residual)
        for destination in range(2):
            result[:,:,destination]=post[:,:,destination,None]*x
            for source in range(2):result[:,:,destination]+=matrix[:,:,source,destination,None]*residual[:,:,source]
        return result
    collapsed,post,matrix=test_glm_flash_hyper.scalar_control(hidden,c.attention_hyper)
    x=norm(collapsed,c.input_norm)
    if sparse:attended=test_glm_flash_sparse.control(x,c.attention.weights,c.attention.indexer)[0]
    else:attended=test_glm_flash_linear.control(x,c.attention)[0]
    residual=restore(attended,hidden,post,matrix)
    collapsed,post,matrix=test_glm_flash_hyper.scalar_control(residual,c.ffn_hyper)
    x=norm(collapsed,c.post_norm);w=c.feed_forward.weights
    gate=(x@w['gate_proj.weight'].T).clamp(max=.5);up=(x@w['up_proj.weight'].T).clamp(-.5,.5)
    fed=(gate*gate.sigmoid()*up)@w['down_proj.weight'].T
    return restore(fed,residual,post,matrix)


class FlashBlockTests(unittest.TestCase):
    def test_complete_equations_both_attention_kinds_and_chunks(self):
        for sparse in (False,True):
            c,x=fixture(sparse);expected=control(x,c,sparse)
            for chunks in ((5,),(2,2,1),(1,1,1,1,1)):
                state=c.new_state(2,5);cursor=0;out=[]
                for size in chunks:
                    out.append(c.forward(x[:,cursor:cursor+size],state,position=cursor));cursor+=size
                torch.testing.assert_close(torch.cat(out,1),expected,atol=5e-7,rtol=5e-5)
                self.assertEqual(state.position,5)
                state.close()

    def test_reorder_and_partial_failure_poisoning(self):
        for sparse in (False,True):
            c,x=fixture(sparse);state=c.new_state(2,5);c.forward(x[:,:2],state,position=0)
            state.reorder(torch.tensor([1,0,1]));expected=control(x,c,sparse)[[1,0,1],2:3]
            torch.testing.assert_close(c.forward(x[[1,0,1],2:3],state,position=2),expected,atol=5e-7,rtol=5e-5)
            with patch.object(FlashFeedForward,'__call__',side_effect=RuntimeError('FFN failure')):
                with self.assertRaisesRegex(RuntimeError,'FFN failure'):c.forward(x[[1,0,1],3:4],state,position=3)
            self.assertTrue(state.poisoned);self.assertEqual(state.position,3);self.assertEqual(state.attention.position,4)
            with self.assertRaises(RuntimeError):state.reorder(torch.tensor([0]))
            with self.assertRaises(RuntimeError):c.forward(x[[1,0,1],3:4],state,position=3)
            state.close();state.close()

    def test_early_failure_retry_owner_cursor_and_budget(self):
        c,x=fixture(False);state=c.new_state(2,5)
        with self.assertRaises(ValueError):c.forward(x+float('nan'),state,position=0)
        self.assertFalse(state.poisoned);self.assertEqual(state.position,0)
        with self.assertRaises(ValueError):fixture(False)[0].forward(x,state,position=0)
        c.max_output_bytes=1
        with patch.object(c.attention,'forward',side_effect=AssertionError('advanced before admission')):
            with self.assertRaises(ValueError):c.forward(x,state,position=0)
        c.max_output_bytes=64*1024**2;c.forward(x[:,:1],state,position=0)
        state.attention.position=2
        with self.assertRaisesRegex(RuntimeError,'cursors'):c.forward(x[:,1:2],state,position=1)
        self.assertTrue(state.poisoned)
