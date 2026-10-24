import unittest
from unittest import mock

from damendaemon import hybrid_q4_policy


def make_torch(arch):
    fake = mock.Mock()
    fake.cuda.get_device_properties.return_value = mock.Mock(gcnArchName=arch)
    return fake


class HybridQ4ContextLimitTest(unittest.TestCase):
    def test_default_limit_is_512_without_env(self):
        with mock.patch.dict('os.environ', {}, clear=False):
            self.assertEqual(hybrid_q4_policy._experimental_context_limit(False), 512)

    def test_chain_limit_unchanged(self):
        self.assertEqual(hybrid_q4_policy._experimental_context_limit(True),
                         hybrid_q4_policy.CHAIN_CONTEXT_LIMIT)

    def test_rocm_gfx1151_opt_in_extends_bound(self):
        with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': '32768'}), \
             mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value='rocm'), \
             mock.patch.dict('sys.modules', {'torch': make_torch('gfx1151')} if False else {}) if False else \
             mock.patch('builtins.__import__', side_effect=lambda name, *a, **k:
                        make_torch('gfx1151') if name == 'torch' else __import__(name, *a, **k)):
            self.assertEqual(hybrid_q4_policy._experimental_context_limit(False), 32768)

    def test_rejects_out_of_bounds_and_garbage(self):
        for value in ('0', '128', '65536', 'abc', '-1'):
            with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': value}), \
                 mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value='rocm'):
                with self.assertRaises(ValueError):
                    hybrid_q4_policy._experimental_context_limit(False)

    def test_rejects_non_rocm_targets(self):
        for accelerator in ('cuda', 'xpu', 'mps'):
            with mock.patch.dict('os.environ', {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': '32768'}), \
                 mock.patch.object(hybrid_q4_policy, 'selected_accelerator', return_value=accelerator):
                with self.assertRaises(ValueError):
                    hybrid_q4_policy._experimental_context_limit(False)


if __name__ == '__main__':
    unittest.main()
