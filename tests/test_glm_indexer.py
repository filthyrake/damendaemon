import unittest
from unittest.mock import patch
import torch
from dllm.glm_indexer import GlmIndexer


def rotate(value,positions,base=8000000.):
    result=value.clone()
    for b in range(value.shape[0]):
        for s in range(value.shape[1]):
            for pair in range(2):
                angle=torch.tensor(float(positions[b,s])/(base**(pair/2)))
                x,y=value[b,s,...,2*pair],value[b,s,...,2*pair+1]
                result[b,s,...,2*pair]=x*angle.cos()-y*angle.sin()
                result[b,s,...,2*pair+1]=x*angle.sin()+y*angle.cos()
    return result


class GlmIndexerTests(unittest.TestCase):
    def setUp(self):
        generator=torch.Generator().manual_seed(71)
        self.weights={name:torch.randn(*shape,generator=generator)*.2 for name,shape in
                      [('wq_b.weight',(16,5)),('wk.weight',(8,6)),('k_norm.weight',(8,)),
                       ('k_norm.bias',(8,)),('weights_proj.weight',(2,6))]}
        self.hidden=torch.randn(2,7,6,generator=generator)
        self.low=torch.randn(2,7,5,generator=generator)
        self.positions=torch.arange(7).repeat(2,1)
        self.indexer=GlmIndexer(self.weights,rope_dim=4)

    def test_key_projection_matches_independent_normalization_and_rotation(self):
        x=self.hidden@self.weights['wk.weight'].T
        centered=x-x.mean(-1,keepdim=True)
        expected=centered/(centered.square().mean(-1,keepdim=True)+1e-6).sqrt()
        expected=expected*self.weights['k_norm.weight']+self.weights['k_norm.bias']
        expected[...,:4]=rotate(expected[...,:4],self.positions)
        for chunk in (1,3,16):
            actual=self.indexer.project_keys(self.hidden,self.positions,chunk=chunk)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_causal_selection_matches_dense_equations_across_chunks(self):
        keys=self.indexer.project_keys(self.hidden,self.positions)
        query=(self.low@self.weights['wq_b.weight'].T).reshape(2,7,2,8)
        query[...,:4]=rotate(query[...,:4],self.positions)
        weights=(self.hidden@self.weights['weights_proj.weight'].T)*(2**-.5)
        dots=torch.einsum('bshd,btd->bsht',query,keys)*(8**-.5)
        dense=(dots.relu()*weights.unsqueeze(-1)).sum(-2)
        dense.masked_fill_(torch.arange(7)[None,None,:]>self.positions[:,:,None],-float('inf'))
        expected_ids=dense.argsort(dim=-1,descending=True,stable=True)[...,:4]
        expected_scores=dense.gather(-1,expected_ids)
        expected_ids=torch.where(expected_ids<=self.positions[:,:,None],expected_ids,-1)
        for q in (1,3,16):
            for k in (1,3,64):
                ids,scores=self.indexer.select(self.hidden,self.low,keys,self.positions,top_k=4,query_chunk=q,key_chunk=k)
                torch.testing.assert_close(ids,expected_ids,rtol=0,atol=0)
                torch.testing.assert_close(scores,expected_scores,rtol=2e-5,atol=2e-6)

    def test_ties_causal_padding_and_different_slot_positions(self):
        keys=torch.zeros(2,7,8)
        positions=torch.tensor([[0,1,2,3,4,5,6],[6,5,4,3,2,1,0]])
        ids,scores=self.indexer.select(self.hidden,self.low,keys,positions,top_k=3,key_chunk=2)
        for b in range(2):
            for s in range(7):
                count=min(3,int(positions[b,s])+1)
                self.assertEqual(ids[b,s].tolist(),list(range(count))+[-1]*(3-count))
                self.assertTrue(torch.isneginf(scores[b,s,count:]).all())

    def test_budget_and_input_rejection_before_projection(self):
        bounded=GlmIndexer(self.weights,rope_dim=4,max_output_bytes=1)
        with patch('dllm.glm_indexer.F.linear',side_effect=AssertionError('projection started')):
            with self.assertRaises(ValueError):bounded.project_keys(self.hidden,self.positions)
            with self.assertRaises(ValueError):bounded.select(self.hidden,self.low,torch.zeros(2,7,8),self.positions,top_k=4)
        for position in (self.positions.float(),self.positions-1,self.positions+2**20):
            with self.assertRaises(ValueError):self.indexer.project_keys(self.hidden,position)
        invalid=self.hidden.clone();invalid[0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.indexer.project_keys(invalid,self.positions)
        with self.assertRaises(ValueError):GlmIndexer(self.weights,rope_dim=3)
        with self.assertRaises(ValueError):self.indexer.select(self.hidden,self.low,torch.zeros(2,7,8),self.positions,top_k=0)
