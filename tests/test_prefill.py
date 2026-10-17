import unittest
from types import SimpleNamespace
import torch
from damendaemon.prefill import groups
from damendaemon.qwen2 import Executor


class PrefillTests(unittest.TestCase):
    def test_bounds_work_and_preserves_order_and_slots(self):
        prompts = [[1]*7, [2]*3, [3]*8, [4]*9, [5]]
        slots = [3,0,4,1,2]
        batches = list(groups(prompts, slots, 10))
        self.assertTrue(all(sum(map(len,batch)) <= 10 for batch,_ in batches))
        self.assertEqual([p for batch,_ in batches for p in batch], prompts)
        self.assertEqual([s for _,assigned in batches for s in assigned], slots)
        self.assertEqual(len(batches), 3)

    def test_small_prompts_stay_batched(self):
        prompts = [[1],[2],[3]]
        self.assertEqual(list(groups(prompts,[2,0,1],20)), [(prompts,[2,0,1])])

    def test_bounded_attention_preserves_global_window_and_tail(self):
        torch.manual_seed(5)
        q = torch.randn(1,2,2071,8)
        k,v = torch.randn(2,1,1,2071,8)
        positions = torch.arange(2071)
        for window in (33,2047):
            mask = (positions[None,:]<=positions[:,None]) & (positions[None,:]>positions[:,None]-window)
            expected = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
            actual = Executor.prefill_attention(SimpleNamespace(window=window,arange=positions),q,k,v)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
