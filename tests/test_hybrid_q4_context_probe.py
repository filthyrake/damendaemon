"""The A30/HOMER context experiment retains device and weight-format bounds."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dllm import hybrid_precision, hybrid_q4_policy
from scripts.experiments.hybrid_q4_context_probe import context_probe


class Q4ContextProbeTests(unittest.TestCase):
    env = {'DLLM_DEVICE': 'cuda', 'DLLM_EXPERIMENTAL_HYBRID_Q4': '1',
           'DLLM_HYBRID_PRECISION': 'compensated', 'DLLM_HYBRID_PREFILL': 'scan'}

    def test_scope_guards_and_original_model_context(self):
        model = SimpleNamespace(config={'model_type': 'qwen3_5_text'},
                                quantization='q4_0', max_context=32768)
        with patch.dict('os.environ', self.env, clear=True), \
                patch('dllm.hybrid_q4_policy.device_name', return_value='NVIDIA A30'):
            with self.assertRaises(ValueError):
                hybrid_q4_policy.validate(model.config, model.max_context)
            with context_probe():
                hybrid_q4_policy.validate(model.config, model.max_context)
                self.assertEqual(hybrid_precision.select(model, False, 4, 'triton'), 'compensated')
                self.assertEqual(model.max_context, 32768)
                for changes in ({'max_context': 511}, {'max_context': 32769},
                                {'quantization': 'q8_0'}, {'vision_weights': object()}):
                    with self.assertRaises(ValueError):
                        hybrid_precision.select(SimpleNamespace(**(vars(model) | changes)), False, 4, 'triton')
                for graph, batch, kernels in ((True, 4, 'triton'), (False, 5, 'triton'), (False, 4, 'torch')):
                    with self.assertRaises(ValueError):
                        hybrid_precision.select(model, graph, batch, kernels)
                for changes in ({'DLLM_DEVICE': 'rocm'}, {'DLLM_DEVICE': 'mps'},
                                {'DLLM_EXPERIMENTAL_HYBRID_Q4': '0'},
                                {'DLLM_HYBRID_PRECISION': 'float16'},
                                {'DLLM_HYBRID_PREFILL': 'fla'}, {'DLLM_EXPERIMENTAL_VISION': '1'}):
                    with patch.dict('os.environ', changes), self.assertRaises(ValueError):
                        hybrid_precision.select(model, False, 4, 'triton')
                with patch('dllm.hybrid_q4_policy.device_name', return_value='NVIDIA A2'), self.assertRaises(ValueError):
                    hybrid_precision.select(model, False, 4, 'triton')
            with self.assertRaises(ValueError):
                hybrid_q4_policy.validate(model.config, model.max_context)

    def test_exception_restores_both_policies(self):
        precision, q4 = hybrid_precision.select, hybrid_q4_policy.validate
        with self.assertRaises(RuntimeError):
            with context_probe():
                raise RuntimeError('stopped probe')
        self.assertIs(hybrid_precision.select, precision)
        self.assertIs(hybrid_q4_policy.validate, q4)

    def test_homer_requires_exact_device_and_preserves_production_context(self):
        model = SimpleNamespace(config={'model_type': 'qwen3_5_text'},
                                quantization='q4_0', max_context=4096)
        with patch.dict('os.environ', self.env | {'DLLM_DEVICE': 'rocm'}, clear=True), \
                patch('dllm.hybrid_q4_policy.device_name', return_value='Radeon 8060S Graphics'):
            with self.assertRaises(ValueError):
                hybrid_precision.select(model, False, 4, 'triton')
            with context_probe():
                self.assertEqual(hybrid_precision.select(model, False, 4, 'triton'), 'compensated')
                self.assertEqual(model.max_context, 4096)
                with patch('dllm.hybrid_q4_policy.device_name', return_value='AMD Instinct MI300X'), self.assertRaises(ValueError):
                    hybrid_precision.select(model, False, 4, 'triton')
            with self.assertRaises(ValueError):
                hybrid_precision.select(model, False, 4, 'triton')
