import unittest
import torch
from damendaemon.glm_flash_attention import project_nope,attend_nope


def fixture():
    torch.manual_seed(950)
    weights={'q_a_proj.weight':torch.randn(3,4)*.2,'q_a_layernorm.weight':torch.randn(3),
             'q_b_proj.weight':torch.randn(6,3)*.2,'kv_a_proj_with_mqa.weight':torch.randn(3,4)*.2,
             'kv_a_layernorm.weight':torch.randn(3)}
    return weights,torch.randn(1,5,4),torch.arange(5).reshape(1,5),torch.randn(10,3)*.2


class FlashAttentionTests(unittest.TestCase):
    def test_nope_projections_and_position_independence(self):
        w,x,pos,_=fixture();actual=project_nope(x,pos,w,heads=2,head_dim=3)
        norm=lambda a,weight:a/torch.sqrt(a.square().mean(-1,keepdim=True)+1e-5)*weight
        low=norm(x@w['q_a_proj.weight'].T,w['q_a_layernorm.weight'])
        expected={'low_rank':low,'query':(low@w['q_b_proj.weight'].T).reshape(1,5,2,3),
                  'latents':norm(x@w['kv_a_proj_with_mqa.weight'].T,w['kv_a_layernorm.weight'])}
        self.assertEqual(set(actual),set(expected))
        shifted=project_nope(x,pos+100,w,heads=2,head_dim=3,chunk=1)
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name])
            torch.testing.assert_close(actual[name],shifted[name])

    def test_selected_tiles_against_dense_masked_attention(self):
        w,x,pos,kv=fixture();p=project_nope(x,pos,w,heads=2,head_dim=3)
        indices=torch.tensor([[[0,-1,-1],[0,1,-1],[0,2,-1],[1,2,3],[0,2,4]]])
        expanded=(p['latents']@kv.T).reshape(1,5,2,5)
        expected=torch.empty(1,5,2,2)
        for t in range(5):
            ids=indices[0,t];ids=ids[ids>=0]
            for h in range(2):
                scores=p['query'][0,t,h]@expanded[0,ids,h,:3].T/(3**.5)
                expected[0,t,h]=scores.softmax(-1)@expanded[0,ids,h,3:]
        for chunk in (1,2,64):
            actual=attend_nope(p['query'],p['latents'],kv,indices,pos,value_dim=2,key_chunk=chunk)
            torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)

    def test_explicit_packed_projection_matches_independently_decoded_control(self):
        from tests.test_glm_packed_attention import packed_pair
        w,x,pos,kv=fixture();packed={};dense={}
        for name,value in w.items():
            if value.ndim==2:packed[name],dense[name]=packed_pair(value)
            else:packed[name]=dense[name]=value
        a=project_nope(x,pos,packed,heads=2,head_dim=3)
        b=project_nope(x,pos,dense,heads=2,head_dim=3)
        for name in a:torch.testing.assert_close(a[name],b[name],atol=2e-6,rtol=2e-5)
        indices=pos.unsqueeze(-1)
        torch.testing.assert_close(attend_nope(a['query'],a['latents'],kv,indices,pos,value_dim=2),
                                   attend_nope(b['query'],b['latents'],kv,indices,pos,value_dim=2),atol=2e-6,rtol=2e-5)

    def test_invalid_selections_nonfinite_and_budgets(self):
        w,x,pos,kv=fixture();p=project_nope(x,pos,w,heads=2,head_dim=3)
        for ids in ([[-1,-1]]*5,[[0,0]]*5,[[4,-1]]*5,[[5,-1]]*5):
            with self.assertRaises(ValueError):attend_nope(p['query'],p['latents'],kv,torch.tensor([ids]),pos,value_dim=2)
        with self.assertRaises(ValueError):project_nope(x*1e30,pos,w,heads=2,head_dim=3)
        with self.assertRaises(ValueError):project_nope(x,pos,w,heads=2,head_dim=3,max_workspace_bytes=1)
        indices=pos.unsqueeze(-1)
        with self.assertRaises(ValueError):attend_nope(p['query']+float('nan'),p['latents'],kv,indices,pos,value_dim=2)
        with self.assertRaises(ValueError):attend_nope(p['query'],p['latents'],kv,indices,pos,value_dim=2,max_output_bytes=1)
