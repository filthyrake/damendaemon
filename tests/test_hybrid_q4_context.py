import unittest
from unittest import mock

from damendaemon import hybrid_q4_policy


class HybridQ4ContextLimitTest(unittest.TestCase):
    def test_default_limit_is_512_without_env(self):
        with mock.patch.dict('os.environ', {}, clear=False):
            self.assertEqual(hybrid_q4_policy._experimental_context_limit(False), 512)

    def test_chain_limit_unchanged(self):
        self.assertEqual(hybrid_q4_policy._experimental_context_limit(True),
                         hybrid_q4_policy.CHAIN_CONTEXT_LIMIT)

    def test_rocm_opt_in_extends_bound(self):
        with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': '32768'}), \
             mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value='rocm'):
            self.assertEqual(hybrid_q4_policy._experimental_context_limit(False), 32768)

    def test_cuda_opt_in_extends_bound_structurally(self):
        # The extension follows admission, not a device or arch string: any
        # admitted accelerator may opt in — the A6000 carries the full 32K
        # evidence set (C1/C2 capacity, lifecycle and cancellation cells), the
        # same bound gfx1151 already exceeded its own 4K qualification under.
        with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': '32768'}), \
             mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value='cuda'):
            self.assertEqual(hybrid_q4_policy._experimental_context_limit(False), 32768)

    def test_rejects_out_of_bounds_and_garbage(self):
        for value in ('0', '128', '65536', 'abc', '-1'):
            with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': value}), \
                 mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value='rocm'):
                with self.assertRaises(ValueError):
                    hybrid_q4_policy._experimental_context_limit(False)

    def test_rejects_unadmitted_targets(self):
        for accelerator in ('xpu', 'mps'):
            with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': '32768'}), \
                 mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value=accelerator):
                with self.assertRaises(ValueError):
                    hybrid_q4_policy._experimental_context_limit(False)


if __name__ == '__main__':
    unittest.main()
