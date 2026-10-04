import unittest
from dllm.prefill import groups


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
