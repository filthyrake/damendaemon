import unittest
from unittest.mock import patch

import torch

from damendaemon.floating_block import FloatingBlock
from damendaemon.hyper_connections import reduce_streams, restore_streams
from damendaemon.floating_attention import attention_prefill
from damendaemon.floating_moe import floating_moe
from damendaemon.expert_selection import select_experts
from tests.test_floating_attention import fixture as attention_fixture


def fixture(hashed=False, ratio=4, dtype=torch.float32, dim=8):
    hidden, attention, freq = attention_fixture(ratio,9,dtype,dim)
    rng = torch.Generator().manual_seed(127)
    weights = {}
    for part in ('attn','ffn'):
        for suffix,shape in (('fn',(8,2*dim)),('base',(8,)),('scale',(3,))):
            weights[f'hc_{part}_{suffix}'] = torch.randn(shape,generator=rng)*.1
        weights[f'{part}_norm.weight'] = torch.randn(dim,generator=rng)*.2
    weights['ffn.gate.weight'] = torch.randn(3,dim,generator=rng)*.2
    if hashed:
        weights['ffn.gate.tid2eid'] = torch.tensor([[0,1],[1,2],[2,0],[0,2]])
    else:
        weights['ffn.gate.bias'] = torch.tensor([.2,-.3,.1])
    for prefix, shapes in (('ffn.experts',((3,6,dim),(3,6,dim),(3,dim,6))),
                           ('ffn.shared_experts',((7,dim),(7,dim),(dim,7)))):
        for name,shape in zip(('w1','w3','w2'),shapes):
            weights[f'{prefix}.{name}.weight'] = (torch.randn(shape,generator=rng)*.2).to(dtype)
    block = FloatingBlock(attention,weights,freq,groups=2,window=4,ratio=ratio,hc_mult=2,top_k=2,index_top_k=2)
    streams = torch.stack((hidden,hidden*.7),dim=2)
    ids = torch.arange(9).remainder(4).unsqueeze(0)
    return block, streams, ids


class FloatingBlockTests(unittest.TestCase):
    def test_full_composition_control(self):
        block,hidden,ids = fixture(False)
        w = block.weights
        def reduce(x,part):
            value,post,matrix = reduce_streams(x,w[f'hc_{part}_fn'],w[f'hc_{part}_scale'],w[f'hc_{part}_base'])
            value = w[f'{part}_norm.weight']*value*(value.square().mean(-1,keepdim=True)+1e-6).rsqrt()
            return value,post,matrix
        reduced,post,matrix = reduce(hidden,'attn')
        attended,_ = attention_prefill(reduced,block.attention_weights,block.frequencies,groups=2,window=4,ratio=4,index_top_k=2)
        intermediate = restore_streams(attended,hidden,post,matrix)
        reduced,post,matrix = reduce(intermediate,'ffn')
        rows = reduced.flatten(0,1)
        coeff,selected = select_experts(rows@w['ffn.gate.weight'].T,2,scoring='sqrtsoftplus',renormalize=True,
                                       correction_bias=w['ffn.gate.bias'],scale=1.5)
        routed = floating_moe(rows,*block.banks,selected.contiguous(),coeff.contiguous(),shared=block.shared,activation_limit=10.)
        expected = restore_streams(routed.reshape(reduced.shape),intermediate,post,matrix)
        actual,_ = block.prefill(hidden,ids)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)

    def test_decode_matches_prefill_for_both_routing_modes(self):
        for hashed in (False,True):
            block,hidden,ids = fixture(hashed)
            _,state = block.prefill(hidden[:,:3],ids[:,:3],max_tokens=9)
            for position in range(3,9):
                actual = block.decode(hidden[:,position:position+1],state,ids[:,position:position+1],position=position)
                expected,_ = block.prefill(hidden[:,:position+1],ids[:,:position+1])
                torch.testing.assert_close(actual,expected[:,-1:],rtol=3e-5,atol=3e-6)

    def test_low_precision_decode_keeps_storage_dtype(self):
        block,hidden,ids = fixture(True,dtype=torch.bfloat16)
        _,state = block.prefill(hidden[:,:3],ids[:,:3],max_tokens=9)
        actual = block.decode(hidden[:,3:4],state,ids[:,3:4],position=3)
        expected,_ = block.prefill(hidden[:,:4],ids[:,:4])
        self.assertEqual(actual.dtype,torch.bfloat16)
        torch.testing.assert_close(actual,expected[:,-1:])

    def test_owner_token_bounds_and_post_attention_failure(self):
        block,hidden,ids = fixture(True)
        other,_,_ = fixture(True)
        _,state = block.prefill(hidden[:,:3],ids[:,:3],max_tokens=9)
        before = state.attention.cache.clone()
        with self.assertRaises(ValueError):
            other.decode(hidden[:,3:4],state,ids[:,3:4],position=3)
        with self.assertRaises((IndexError,RuntimeError)):
            block.decode(hidden[:,3:4],state,torch.tensor([[99]]),position=3)
        torch.testing.assert_close(state.attention.cache,before)
        self.assertEqual(state.attention.position,3)
        with patch('damendaemon.floating_block.floating_moe',side_effect=RuntimeError('expert failure')):
            with self.assertRaisesRegex(RuntimeError,'expert failure'):
                block.decode(hidden[:,3:4],state,ids[:,3:4],position=3)
        self.assertTrue(state.attention.poisoned)
        self.assertEqual(state.attention.position,4)
        with self.assertRaisesRegex(RuntimeError,'poisoned'):
            block.decode(hidden[:,4:5],state,ids[:,4:5],position=4)
        state.close()
        self.assertIsNone(state.attention.cache)

    def test_reordered_slots_and_parameter_validation(self):
        block,hidden,ids = fixture(True)
        _,state = block.prefill(hidden[:,:3],ids[:,:3],max_tokens=9)
        state.reorder(torch.tensor([0,0]))
        actual = block.decode(hidden[:,3:4].expand(2,-1,-1,-1),state,ids[:,3:4].expand(2,-1),position=3)
        expected,_ = block.prefill(hidden[:,:4],ids[:,:4])
        torch.testing.assert_close(actual,expected[:,-1:].expand(2,-1,-1,-1),rtol=3e-5,atol=3e-6)
        bad = dict(block.weights)
        bad['ffn.gate.tid2eid'] = torch.tensor([[0,0]])
        with self.assertRaises(ValueError):
            FloatingBlock(block.attention_weights,bad,block.frequencies,groups=2,window=4,ratio=4,hc_mult=2,top_k=2)
