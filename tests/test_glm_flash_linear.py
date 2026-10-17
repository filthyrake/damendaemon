import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from damendaemon.glm_flash_linear import FlashLinearAttention
from tests import test_glm_flash_kda


def layer(kernel=4,**options):
    torch.manual_seed(819)
    shapes={f'{n}_proj.weight':(6,4) for n in ('q','k','v')}
    shapes.update({f'{n}_conv1d.weight':(6,1,kernel) for n in ('q','k','v')})
    shapes.update({f'{n}_a_proj.weight':(3,4) for n in ('f','g')})
    shapes.update({f'{n}_b_proj.weight':(6,3) for n in ('f','g')})
    shapes.update({'b_proj.weight':(2,4),'o_proj.weight':(4,6),'o_norm.weight':(3,),'A_log':(2,),'dt_bias':(6,)})
    weights={n:torch.randn(shape)*.3 for n,shape in shapes.items()}
    return FlashLinearAttention(weights,heads=2,head_dim=3,kernel_size=kernel,**options)


def control(x,c):
    w=c.weights;b,t,_=x.shape
    linear=lambda x,n:x@w[n+'.weight'].T
    qkv=[]
    for n in ('q','k','v'):
        raw=linear(x,n+'_proj');out=torch.zeros_like(raw)
        for token in range(t):
            for tap in range(c.kernel_size):
                source=token+tap-c.kernel_size+1
                if source>=0:out[:,token]+=raw[:,source]*w[n+'_conv1d.weight'][:,0,tap]
        qkv.append(F.silu(out).reshape(b,t,2,3))
    forget=linear(linear(x,'f_a_proj'),'f_b_proj')+w['dt_bias']
    decay=-5*torch.sigmoid(forget.reshape(b,t,2,3)*w['A_log'].exp().reshape(1,1,2,1))
    beta=linear(x,'b_proj').sigmoid()
    core,state=test_glm_flash_kda.scalar_control(*qkv,decay,beta,torch.zeros(b,2,3,3))
    gate=linear(linear(x,'g_a_proj'),'g_b_proj').reshape(b,t,2,3)
    output=core/torch.sqrt(core.square().mean(-1,keepdim=True)+c.epsilon)*w['o_norm.weight']*gate.sigmoid()
    return linear(output.reshape(b,t,6),'o_proj'),state


class FlashLinearTests(unittest.TestCase):
    def test_complete_layer_independent_control_and_chunk_continuation(self):
        for kernel in (1,4):
            c=layer(kernel);x=torch.randn(2,5,4);expected,expected_state=control(x,c)
            whole=c.new_state(2,8);actual=c.forward(x,whole,position=0)
            torch.testing.assert_close(actual,expected,atol=3e-7,rtol=3e-5)
            torch.testing.assert_close(whole.recurrent,expected_state,atol=2e-7,rtol=3e-6)
            for chunks in ((2,2,1),(1,1,1,1,1)):
                state=c.new_state(2,8);cursor=0;outputs=[]
                for size in chunks:
                    outputs.append(c.forward(x[:,cursor:cursor+size],state,position=cursor));cursor+=size
                torch.testing.assert_close(torch.cat(outputs,1),actual,atol=3e-7,rtol=3e-5)
                torch.testing.assert_close(state.convolution,whole.convolution)
                torch.testing.assert_close(state.recurrent,whole.recurrent,atol=2e-7,rtol=3e-6)

    def test_reorder_shrink_duplicate_and_release(self):
        c=layer();x=torch.randn(2,4,4);state=c.new_state(2,8)
        c.forward(x[:,:3],state,position=0);state.reorder(torch.tensor([1,0,1]))
        actual=c.forward(x[[1,0,1],3:],state,position=3)
        expected=control(x,c)[0][[1,0,1],3:]
        torch.testing.assert_close(actual,expected,atol=3e-7,rtol=3e-5)
        state.reorder(torch.tensor([2]));self.assertEqual(state.recurrent.shape[0],1)
        state.close();state.close();self.assertIsNone(state.recurrent)
        with self.assertRaises(RuntimeError):c.forward(x[:1,:1],state,position=4)
        fresh=c.new_state(1,8);torch.testing.assert_close(c.forward(x[:1],fresh,position=0),control(x[:1],c)[0],atol=3e-7,rtol=3e-5)

    def test_failure_preserves_cache_and_cursor(self):
        c=layer();x=torch.randn(1,2,4);state=c.new_state(1,8);c.forward(x,state,position=0)
        old=(state.convolution.clone(),state.recurrent.clone())
        with patch('damendaemon.glm_flash_linear.recurrent_attention',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):c.forward(x,state,position=2)
        self.assertEqual(state.position,2)
        torch.testing.assert_close(state.convolution,old[0]);torch.testing.assert_close(state.recurrent,old[1])
        c.forward(x,state,position=2);self.assertEqual(state.position,4)

    def test_admission_and_foreign_state(self):
        c=layer();x=torch.randn(1,2,4);state=c.new_state(1,2)
        with self.assertRaises(ValueError):layer().forward(x,state,position=0)
        with self.assertRaises(ValueError):c.new_state(1,8,max_cache_bytes=1)
        with self.assertRaises(ValueError):c.forward(x,state,position=1)
        with self.assertRaises(ValueError):c.forward(x+float('nan'),state,position=0)
        self.assertEqual(state.position,0)
        for option in ('max_workspace_bytes','max_output_bytes'):
            small=layer(**{option:1});ss=small.new_state(1,2)
            with patch('damendaemon.glm_flash_linear._finite',side_effect=AssertionError('scanned before admission')):
                with self.assertRaisesRegex(ValueError,'budget'):small.forward(x,ss,position=0)
        c.forward(x,state,position=0)
        with self.assertRaises(ValueError):c.forward(x[:,:1],state,position=2)
        with self.assertRaises(ValueError):state.reorder(torch.tensor([5]))
