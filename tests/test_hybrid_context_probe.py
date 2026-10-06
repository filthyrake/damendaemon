"""The context experiment must retain every other production admission guard."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dllm import hybrid_precision
from scripts.experiments.hybrid_context_probe import context_probe


class HybridContextProbeTests(unittest.TestCase):
    def test_only_context_scope_changes_and_policy_is_restored(self):
        model = SimpleNamespace(max_context=32768, quantization='q8_0',
                                config={'model_type': 'qwen3_5_text'})
        original = hybrid_precision.select
        env = {'DLLM_DEVICE': 'rocm', 'DLLM_HYBRID_PRECISION': 'compensated',
               'DLLM_HYBRID_PREFILL': 'scan'}
        with patch.dict('os.environ', env, clear=True):
            with self.assertRaises(ValueError):
                original(model, False, 4, 'triton')
            with context_probe():
                self.assertEqual(hybrid_precision.select(model, False, 4, 'triton'), 'compensated')
                self.assertEqual(model.max_context, 32768)
                for graph, batch, kernels in ((True, 4, 'triton'), (False, 5, 'triton'), (False, 4, 'torch')):
                    with self.assertRaises(ValueError):
                        hybrid_precision.select(model, graph, batch, kernels)
                for field, value in (('max_context', 32769), ('max_context', 511),
                                     ('quantization', 'q4_0'), ('vision_weights', {})):
                    with patch.object(model, field, value, create=True):
                        with self.assertRaises(ValueError):
                            hybrid_precision.select(model, False, 4, 'triton')
                for overrides in ({'DLLM_DEVICE': 'cuda'}, {'DLLM_HYBRID_PRECISION': 'float16'}):
                    with patch.dict('os.environ', overrides):
                        with self.assertRaises(ValueError):
                            hybrid_precision.select(model, False, 4, 'triton')
            self.assertIs(hybrid_precision.select, original)

    def test_exception_restores_policy(self):
        original = hybrid_precision.select
        with self.assertRaises(RuntimeError):
            with context_probe():
                raise RuntimeError('failed probe')
        self.assertIs(hybrid_precision.select, original)
