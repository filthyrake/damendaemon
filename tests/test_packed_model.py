import unittest
from unittest.mock import patch

import torch

from dllm.floating_block import FloatingBlock
from dllm.floating_model import FloatingModel
from tests.test_floating_block import fixture as block_fixture
from tests.test_packed_moe import expert


def fixture():
    experts, dense = zip(*(expert(format) for format in ('e2m1','e4m3fn','e2m1')))
    shared,shared_dense = expert('e4m3fn',intermediate=32)
    packed_blocks,controls = [],[]
    for hashed,ratio in ((True,0),(False,4)):
        base,_,ids = block_fixture(hashed,ratio,dim=32)
        weights = {name:value for name,value in base.weights.items()
                   if not name.startswith(('ffn.experts.','ffn.shared_experts.'))}
        options = dict(groups=2,window=4,ratio=ratio,hc_mult=2,top_k=2,index_top_k=2,activation_limit=2)
        packed_blocks.append(FloatingBlock(base.attention_weights,weights,base.frequencies,
                                           packed_experts=experts,packed_shared=shared,**options))
        floating = dict(weights)
        for index,projection in enumerate(('w1','w3','w2')):
            floating[f'ffn.experts.{projection}.weight'] = torch.stack([values[index] for values in dense])
            floating[f'ffn.shared_experts.{projection}.weight'] = shared_dense[index]
        controls.append(FloatingBlock(base.attention_weights,floating,base.frequencies,**options))
    rng = torch.Generator().manual_seed(163)
    embedding = torch.randn(4,32,generator=rng)*.2
    shapes = {'head.weight':(4,32),'norm.weight':(32,), 'hc_head_fn':(2,64),'hc_head_scale':(1,),'hc_head_base':(2,)}
    head = {name:torch.randn(shape,generator=rng)*.2 for name,shape in shapes.items()}
    return FloatingModel(embedding,packed_blocks,head),FloatingModel(embedding,controls,head),ids


class PackedModelTests(unittest.TestCase):
    def test_complete_model_against_scalar_decoded_floating_experts(self):
        packed,control,ids = fixture()
        self.assertTrue(all(b.banks is None and b.shared is None for b in packed.blocks))
        self.assertTrue(all(not any(n.startswith('ffn.experts.') for n in b.weights) for b in packed.blocks))
        expected,_ = control.prefill(ids[:,:3])
        with patch('dllm.floating_block.floating_moe',side_effect=AssertionError('floating expert fallback')):
            actual,state = packed.prefill(ids[:,:3],max_tokens=7)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        for position in range(3,7):
            actual = packed.decode(ids[:,position:position+1],state,position=position)
            expected,_ = control.prefill(ids[:,:position+1])
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        state.close()

    def test_slot_fanout_and_packed_expert_failure_poison_model(self):
        packed,control,ids = fixture()
        _,state = packed.prefill(ids[:,:3],max_tokens=7)
        state.reorder(torch.tensor([0,0]))
        actual = packed.decode(torch.tensor([[0],[3]]),state,position=3)
        expected,_ = control.prefill(torch.cat((ids[:,:3].expand(2,-1),torch.tensor([[0],[3]])),1))
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        with patch('dllm.floating_block.packed_moe',side_effect=RuntimeError('packed failure')):
            with self.assertRaisesRegex(RuntimeError,'packed failure'):
                packed.decode(torch.tensor([[1],[1]]),state,position=4)
        self.assertTrue(state.poisoned)
        self.assertTrue(all(s.attention.poisoned for s in state.layers))
        state.close()

    def test_explicit_packed_contract_rejects_mismatches(self):
        model,_,_ = fixture()
        b = model.blocks[0]
        options = dict(groups=2,window=4,hc_mult=2,top_k=2,activation_limit=2,
                       packed_experts=b.packed_experts,packed_shared=b.packed_shared)
        for override in ({'packed_experts':b.packed_experts[:2]}, {'packed_shared':None},
                         {'activation_limit':10}, {'workspace_bytes':1}, {'packed_experts':None}):
            with self.assertRaises(ValueError):
                FloatingBlock(b.attention_weights,b.weights,b.frequencies,**(options|override))
        extra = dict(b.weights)
        extra['ffn.experts.w1.weight'] = torch.zeros(3,64,32)
        with self.assertRaises(ValueError):
            FloatingBlock(b.attention_weights,extra,b.frequencies,**options)
