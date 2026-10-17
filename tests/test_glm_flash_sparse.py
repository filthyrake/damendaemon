import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_sparse import FlashSparseAttention
from tests import test_glm_flash_indexer


def fixture(packed=False):
    index=test_glm_flash_indexer.indexer();torch.manual_seed(981)
    shapes={'q_a_proj.weight':(2,4),'q_a_layernorm.weight':(2,),'q_b_proj.weight':(6,2),
            'kv_a_proj_with_mqa.weight':(3,4),'kv_a_layernorm.weight':(3,),'kv_b_proj.weight':(10,3),'o_proj.weight':(4,4)}
    weights={n:torch.randn(s)*.3 for n,s in shapes.items()};dense=dict(weights)
    if packed:
        from tests.test_glm_packed_attention import packed_pair
        for name in ('q_a_proj.weight','q_b_proj.weight','kv_a_proj_with_mqa.weight','o_proj.weight'):
            weights[name],dense[name]=packed_pair(weights[name])
    return FlashSparseAttention(weights,index,heads=2,head_dim=3,value_dim=2),dense,torch.randn(2,7,4)


def control(x,w,index):
    norm=lambda a,scale:a/torch.sqrt(a.square().mean(-1,keepdim=True)+1e-5)*scale
    b,t,_=x.shape;low=norm(x@w['q_a_proj.weight'].T,w['q_a_layernorm.weight'])
    q=(low@w['q_b_proj.weight'].T).reshape(b,t,2,3)
    latents=norm(x@w['kv_a_proj_with_mqa.weight'].T,w['kv_a_layernorm.weight'])
    kv=(latents@w['kv_b_proj.weight'].T).reshape(b,t,2,5)
    keys,gates=index.project_keys(x);ids=index.select(x,low,keys,gates,torch.arange(t).expand(b,-1))
    output=torch.empty(b,t,2,2)
    for batch in range(b):
        for token in range(t):
            selected=ids[batch,token];selected=selected[selected>=0]
            for head in range(2):
                logits=q[batch,token,head]@kv[batch,selected,head,:3].T/(3**.5)
                output[batch,token,head]=logits.softmax(-1)@kv[batch,selected,head,3:]
    return output.reshape(b,t,4)@w['o_proj.weight'].T,ids


class FlashSparseTests(unittest.TestCase):
    def test_dense_control_chunk_decode_and_packed_projections(self):
        for packed in (False,True):
            c,w,x=fixture(packed);expected,ids=control(x,w,c.indexer)
            for chunks in ((7,),(1,1,1,1,1,1,1),(3,2,2)):
                state=c.new_state(2,8);cursor=0;out=[];selected=[]
                for size in chunks:
                    y,i=c.forward(x[:,cursor:cursor+size],state,position=cursor);out.append(y);selected.append(i);cursor+=size
                torch.testing.assert_close(torch.cat(out,1),expected,atol=3e-7,rtol=3e-5)
                torch.testing.assert_close(torch.cat(selected,1),ids,atol=0,rtol=0)
                self.assertEqual(state.position,7)

    def test_reorder_release_and_fresh_state(self):
        c,w,x=fixture();state=c.new_state(2,8);c.forward(x[:,:3],state,position=0)
        state.reorder(torch.tensor([1,0,1]));state.reorder(torch.tensor([2]))
        actual,_=c.forward(x[1:2,3:],state,position=3)
        torch.testing.assert_close(actual,control(x[1:2],w,c.indexer)[0][:,3:],atol=3e-7,rtol=3e-5)
        state.close();state.close();self.assertIsNone(state.latents)
        with self.assertRaises(RuntimeError):c.forward(x[:1,:1],state,position=7)
        fresh=c.new_state(2,8);torch.testing.assert_close(c.forward(x,fresh,position=0)[0],control(x,w,c.indexer)[0],atol=3e-7,rtol=3e-5)

    def test_late_failure_is_transactional_and_retryable(self):
        c,w,x=fixture();state=c.new_state(2,8);c.forward(x[:,:3],state,position=0)
        old=[v.clone() for v in (state.latents,state.keys,state.gates)]
        with patch('damendaemon.glm_flash_sparse.linear',side_effect=RuntimeError('late projection')):
            with self.assertRaisesRegex(RuntimeError,'late projection'):c.forward(x[:,3:],state,position=3)
        self.assertEqual(state.position,3)
        for a,b in zip((state.latents,state.keys,state.gates),old):torch.testing.assert_close(a,b)
        actual,_=c.forward(x[:,3:],state,position=3)
        torch.testing.assert_close(actual,control(x,w,c.indexer)[0][:,3:],atol=3e-7,rtol=3e-5)

    def test_admission_bounds_and_owner(self):
        c,w,x=fixture();state=c.new_state(2,7)
        with self.assertRaises(ValueError):fixture()[0].forward(x,state,position=0)
        with self.assertRaises(ValueError):c.new_state(2,7,max_cache_bytes=1)
        with self.assertRaises(ValueError):c.forward(x,state,position=1)
        c.max_workspace_bytes=1
        with patch('damendaemon.glm_flash_sparse.project_nope',side_effect=AssertionError('projection before admission')):
            with self.assertRaises(ValueError):c.forward(x,state,position=0)
        self.assertEqual(state.position,0)
