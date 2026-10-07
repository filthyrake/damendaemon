import unittest
from unittest.mock import patch
import torch
from dllm.glm_attention import project_mla,sparse_latent_attention
from tests.test_glm_indexer import rotate


class GlmAttentionTests(unittest.TestCase):
    def setUp(self):
        generator=torch.Generator().manual_seed(17)
        self.weights={name:torch.randn(*shape,generator=generator)*.2 for name,shape in (
            ('q_a_proj.weight',(5,6)),('q_a_layernorm.weight',(5,)),('q_b_proj.weight',(12,5)),
            ('kv_a_proj_with_mqa.weight',(7,6)),('kv_a_layernorm.weight',(3,)))}
        self.hidden=torch.randn(2,7,6,generator=generator)
        self.positions=torch.arange(7).repeat(2,1)
        self.kv=torch.randn(10,3,generator=generator)*.2

    def projected(self,**kwargs):
        return project_mla(self.hidden,self.positions,self.weights,heads=2,nope_dim=2,rope_dim=4,**kwargs)

    def test_projections_match_dense_equations_and_tail_rotation(self):
        w=self.weights
        def rms(value,weight):return value/(value.square().mean(-1,keepdim=True)+1e-5).sqrt()*weight
        low=rms(self.hidden@w['q_a_proj.weight'].T,w['q_a_layernorm.weight'])
        query=(low@w['q_b_proj.weight'].T).reshape(2,7,2,6)
        query[...,2:]=rotate(query[...,2:],self.positions)
        raw=self.hidden@w['kv_a_proj_with_mqa.weight'].T
        expected={'query':query,'low_rank':low,'latents':rms(raw[...,:3],w['kv_a_layernorm.weight']),
                  'rotary_keys':rotate(raw[...,3:],self.positions)}
        for chunk in (1,3,16):
            actual=self.projected(chunk=chunk)
            for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=2e-5,atol=2e-6)

    def test_sparse_tiled_attention_matches_full_expanded_masked_control(self):
        p=self.projected()
        indices=torch.full((2,7,4),-1,dtype=torch.int64)
        for b in range(2):
            for s in range(7):
                chosen=list(range(s,-1,-2))[:4]
                indices[b,s,:len(chosen)]=torch.tensor(chosen)
        expanded=(p['latents']@self.kv.T).reshape(2,7,2,5)
        key=torch.cat((expanded[...,:2],p['rotary_keys'][:,:,None,:].expand(-1,-1,2,-1)),-1)
        scores=torch.einsum('bshd,bthd->bsht',p['query'],key)*(6**-.5)
        mask=torch.ones(2,7,7,dtype=torch.bool)
        for b in range(2):
            for s in range(7):mask[b,s,indices[b,s][indices[b,s]>=0]]=False
        scores.masked_fill_(mask[:,:,None,:],-float('inf'))
        expected=torch.einsum('bsht,bthv->bshv',scores.softmax(-1),expanded[...,2:])
        original=p['latents'].clone()
        for chunk in (1,2,64):
            actual=sparse_latent_attention(p['query'],p['latents'],p['rotary_keys'],self.kv,indices,self.positions,
                                            nope_dim=2,value_dim=3,key_chunk=chunk)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(p['latents'],original,rtol=0,atol=0)

    def test_invalid_selection_and_budgets(self):
        p=self.projected();valid=torch.zeros(2,7,1,dtype=torch.int64)
        def attend(ids=valid,**kwargs):
            return sparse_latent_attention(p['query'],p['latents'],p['rotary_keys'],self.kv,ids,self.positions,
                                           nope_dim=2,value_dim=3,**kwargs)
        for bad in (torch.full_like(valid,-1),torch.full_like(valid,-2),torch.full_like(valid,7),
                    torch.ones_like(valid),valid.expand(2,7,2)):
            with self.assertRaises(ValueError):attend(bad)
        with patch('dllm.glm_attention.F.linear',side_effect=AssertionError('projection started')):
            with self.assertRaises(ValueError):attend(max_workspace_bytes=1)
            with self.assertRaises(ValueError):self.projected(max_output_bytes=1)
        p['query'][0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):attend()
