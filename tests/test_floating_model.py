import unittest
from unittest.mock import patch

import torch

from dllm.floating_model import FloatingModel
from dllm.hyper_connections import reduce_head
from tests.test_floating_block import fixture as block_fixture


def fixture():
    blocks = [block_fixture(True,0)[0],block_fixture(False,4)[0]]
    rng = torch.Generator().manual_seed(131)
    embedding = torch.randn(4,8,generator=rng)*.2
    shapes = {'head.weight':(4,8),'norm.weight':(8,), 'hc_head_fn':(2,16),'hc_head_scale':(1,),'hc_head_base':(2,)}
    head = {name:torch.randn(shape,generator=rng)*.2 for name,shape in shapes.items()}
    return FloatingModel(embedding,blocks,head),torch.tensor([[0,1,2,3,0,2,1,3,2]])


class FloatingModelTests(unittest.TestCase):
    def test_head_composition_teacher_forcing_and_greedy_steps(self):
        model,ids = fixture()
        actual,_ = model.prefill(ids[:,:3])
        hidden = model.embedding[ids[:,:3]].unsqueeze(2).expand(-1,-1,2,-1)
        for block in model.blocks:
            hidden,_ = block.prefill(hidden,ids[:,:3])
        h = model.head
        reduced = reduce_head(hidden[:,-1:],h['hc_head_fn'],h['hc_head_scale'],h['hc_head_base'])
        norm = h['norm.weight']*(reduced*(reduced.square().mean(-1,keepdim=True)+1e-6).rsqrt())
        torch.testing.assert_close(actual,norm[:,0]@h['head.weight'].T)
        _,state = model.prefill(ids[:,:3],max_tokens=9)
        for position in range(3,9):
            actual = model.decode(ids[:,position:position+1],state,position=position)
            expected,_ = model.prefill(ids[:,:position+1])
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        sequence = ids[:,:3]
        logits,state = model.prefill(sequence,max_tokens=7)
        for position in range(3,7):
            token = logits.argmax(-1,keepdim=True)
            sequence = torch.cat((sequence,token),1)
            logits = model.decode(token,state,position=position)
            expected,_ = model.prefill(sequence)
            torch.testing.assert_close(logits,expected,rtol=3e-5,atol=3e-6)

    def test_reordered_slots_and_global_budget(self):
        model,ids = fixture()
        _,state = model.prefill(ids[:,:3],max_tokens=9)
        state.reorder(torch.tensor([0,0]))
        actual = model.decode(ids[:,3:4].expand(2,-1),state,position=3)
        expected,_ = model.prefill(ids[:,:4])
        torch.testing.assert_close(actual,expected.expand(2,-1),rtol=3e-5,atol=3e-6)
        state.reorder(torch.tensor([1]))
        self.assertEqual(state.layers[0].attention.cache.shape[0],1)
        per_slot = sum(layer.attention.bytes_per_slot for layer in state.layers)
        state.max_cache_bytes = per_slot
        with self.assertRaises(ValueError):
            state.reorder(torch.tensor([0,0]))
        with self.assertRaises(ValueError):
            model.prefill(ids[:,:3],max_tokens=9,max_cache_bytes=per_slot-1)
        self.assertFalse(model.failed_prefills)

    def test_late_layer_and_head_failures_poison_entire_model(self):
        for where in ('layer','head'):
            model,ids = fixture()
            _,state = model.prefill(ids[:,:3],max_tokens=9)
            target, method = (model.blocks[1],'decode') if where == 'layer' else (model,'_logits')
            with patch.object(target,method,side_effect=RuntimeError('injected failure')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    model.decode(ids[:,3:4],state,position=3)
            self.assertTrue(state.poisoned)
            self.assertTrue(all(layer.attention.poisoned for layer in state.layers))
            self.assertEqual(state.position,3)
            with self.assertRaisesRegex(RuntimeError,'poisoned'):
                model.decode(ids[:,3:4],state,position=3)
            state.close(); state.close()
            self.assertFalse(state.layers)

    def test_reorder_partial_failure_and_layer_identity(self):
        model,ids = fixture()
        for failed_layer in (0,1):
            _,state = model.prefill(ids[:,:3],max_tokens=9)
            with patch.object(state.layers[failed_layer],'reorder',side_effect=RuntimeError('copy failed')):
                with self.assertRaisesRegex(RuntimeError,'copy failed'):
                    state.reorder(torch.tensor([0,0]))
            self.assertEqual(state.poisoned,failed_layer == 1)
            state.close()
        for corruption in ('missing','foreign'):
            _,state = model.prefill(ids[:,:3],max_tokens=9)
            original = list(state.layers)
            if corruption == 'missing':
                state.layers.pop()
            else:
                state.layers[1].owner = model.blocks[0]
            with self.assertRaisesRegex(RuntimeError,'disagree'):
                model.decode(ids[:,3:4],state,position=3)
            self.assertTrue(state.poisoned)
            self.assertTrue(all(s.attention.position == 3 for s in original))
            for layer in original:
                layer.close()

    def test_exact_aggregate_budget_allows_fanout(self):
        model,ids = fixture()
        _,sample = model.prefill(ids[:,:3],max_tokens=9)
        budget = 2*sum(s.attention.bytes_per_slot for s in sample.layers)
        sample.close()
        _,state = model.prefill(ids[:,:3],max_tokens=9,max_cache_bytes=budget)
        state.reorder(torch.tensor([0,0]))
        self.assertTrue(all(s.attention.cache.shape[0] == 2 for s in state.layers))
        with self.assertRaises(ValueError):
            state.reorder(torch.tensor([0,0,1]))
        self.assertFalse(state.poisoned)

    def test_invalid_token_stale_position_and_owner_preserve_cache(self):
        model,ids = fixture()
        other,_ = fixture()
        _,state = model.prefill(ids[:,:3],max_tokens=9)
        with self.assertRaises(ValueError):
            other.decode(ids[:,3:4],state,position=3)
        with self.assertRaises(ValueError):
            model.decode(ids[:,3:4],state,position=2)
        with self.assertRaises((IndexError,RuntimeError)):
            model.decode(torch.tensor([[-1]]),state,position=3)
        self.assertEqual(state.position,3)
        self.assertTrue(all(s.attention.position == 3 for s in state.layers))
        self.assertFalse(state.poisoned)

    def test_failed_prefill_cleanup_retains_owner_and_retries_before_rebuild(self):
        model,ids = fixture()
        captured = []
        original = model.blocks[0].prefill
        def prepare(*args,**kwargs):
            output,state = original(*args,**kwargs)
            captured.append(state)
            close = state.close
            first = True
            def flaky_close():
                nonlocal first
                if first:
                    first = False
                    raise RuntimeError('cleanup failed')
                close()
            state.close = flaky_close
            return output,state
        with patch.object(model.blocks[0],'prefill',side_effect=prepare), \
                patch.object(model.blocks[1],'prefill',side_effect=RuntimeError('construction failed')):
            with self.assertRaisesRegex(RuntimeError,'construction failed'):
                model.prefill(ids[:,:3])
        self.assertEqual(len(model.failed_prefills),1)
        self.assertIsNotNone(captured[0].attention.cache)
        model.prefill(ids[:,:3])
        self.assertFalse(model.failed_prefills)
        self.assertIsNone(captured[0].attention.cache)
