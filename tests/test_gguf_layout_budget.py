import unittest

from damendaemon.pages import PagePool
from scripts.verify_gguf_layout_runtime import paged_capacity


class LayoutBudgetTests(unittest.TestCase):
    def test_all_slots_cross_page_boundaries_through_final_decode(self):
        for context, lengths in ((4104, (4, 4095, 4096, 4097)),
                                 (512, (4, 128, 256, 384)),
                                 (512, (505, 505, 505, 505))):
            with self.subTest(context=context, lengths=lengths):
                pool = PagePool(paged_capacity(context, lengths), 4, context)
                for step in range(8):
                    pool.resize_many([(slot, length+step) for slot, length in enumerate(lengths)])
                self.assertGreater(pool.used_tokens, 0)
                for slot in range(4):
                    pool.release(slot)
                self.assertEqual(pool.used_tokens, 0)


if __name__ == '__main__':
    unittest.main()
