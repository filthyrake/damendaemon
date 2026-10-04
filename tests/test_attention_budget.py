import unittest
from dllm.attention_budget import query_rows


class AttentionBudgetTests(unittest.TestCase):
    def test_storage_fits_for_short_and_long_context_split_counts(self):
        budget=32*1024**2
        for splits in (1,2,4,16,32,512):
            rows=query_rows(64,64,splits,budget)
            self.assertLessEqual(rows*64*(splits*65+64)*4,budget)
            self.assertGreater((rows+1)*64*(splits*65+64)*4,budget)
        self.assertEqual(query_rows(64,64,16,budget),118)
        self.assertEqual(query_rows(64,64,32,budget),61)

    def test_bad_or_unfulfillable_budget_is_explicit(self):
        for budget in (None,0,-1,True,1.5,64*16*65*4-1):
            with self.subTest(budget=budget),self.assertRaises(ValueError):query_rows(64,64,16,budget)


    def test_output_dtype_is_part_of_the_workspace_bound(self):
        budget=32*1024**2
        for itemsize in (2,4,8):
            rows=query_rows(64,64,16,budget,itemsize)
            per_row=64*(16*65*4+64*itemsize)
            self.assertLessEqual(rows*per_row,budget)
            self.assertGreater((rows+1)*per_row,budget)
