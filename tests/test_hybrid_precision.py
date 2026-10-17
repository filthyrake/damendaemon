"""Precision admission must reject unqualified combinations before GPU allocation."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from damendaemon.hybrid_precision import select


class HybridPrecisionTests(unittest.TestCase):
    def model(self, **kwargs):
        return SimpleNamespace(**(dict(quantization='q8_0', max_context=512,
                                       config={'model_type': 'qwen3_5_text'}) | kwargs))

    def test_default_does_not_change_existing_admission(self):
        with patch.dict('os.environ', {'DAMENDAEMON_HYBRID_PRECISION': 'float16', 'DAMENDAEMON_DEVICE': 'mps'}):
            self.assertEqual(select(self.model(max_context=32768), True, 16, 'torch'), 'float16')

    def test_explicit_modes_have_bounded_scope(self):
        for mode in ('float32', 'compensated'):
            with patch.dict('os.environ', {'DAMENDAEMON_HYBRID_PRECISION': mode, 'DAMENDAEMON_DEVICE': 'rocm', 'DAMENDAEMON_HYBRID_PREFILL': 'scan'}, clear=True):
                self.assertEqual(select(self.model(), False, 4, 'triton'), mode)
                for model, graph, batch, kernels in (
                    (self.model(max_context=513), False, 4, 'triton'),
                    (self.model(quantization='q4_0'), False, 4, 'triton'),
                    (self.model(config={'model_type': 'qwen3'}), False, 4, 'triton'),
                    (self.model(vision_weights={}), False, 4, 'triton'),
                    (self.model(), True, 4, 'triton'),
                    (self.model(), False, 5, 'triton'),
                    (self.model(), False, 4, 'torch')):
                    with self.subTest(mode=mode, model=model, graph=graph, batch=batch, kernels=kernels):
                        with self.assertRaises(ValueError):
                            select(model, graph, batch, kernels)
                for overrides in ({'DAMENDAEMON_DEVICE': 'cuda'}, {'DAMENDAEMON_DEVICE': 'mps'}, {'DAMENDAEMON_HYBRID_PREFILL': 'fla'}):
                    with patch.dict('os.environ', overrides):
                        with self.assertRaises(ValueError):
                            select(self.model(), False, 4, 'triton')

    def test_unknown_mode_rejected(self):
        with patch.dict('os.environ', {'DAMENDAEMON_HYBRID_PRECISION': 'automatic'}):
            with self.assertRaises(ValueError):
                select(self.model(), False, 4, 'triton')

    def test_q8_callback_can_select_or_decline_without_changing_other_weights(self):
        from damendaemon.quantization import Linear
        weights = {'q': torch.zeros((4, 32), dtype=torch.int8), 'plain': torch.ones((4, 32))}
        scales = {'q': torch.ones((4, 1), dtype=torch.float16)}
        calls = []
        def projection(x, weight, scale, bias):
            calls.append((weight, scale, bias))
            return torch.full((x.shape[0], 4), 7.) if x.shape[0] >= 2 else None
        fallback = lambda x, weight, bias: torch.nn.functional.linear(x, weight, bias)
        linear = Linear(weights, scales, 'cpu', fallback, q8_projection=projection)
        self.assertTrue(torch.equal(linear(torch.ones(2, 32), weights['q']), torch.full((2, 4), 7.)))
        self.assertIs(calls[-1][0], weights['q'])
        self.assertTrue(torch.equal(calls[-1][1], scales['q']))
        count = len(calls)
        self.assertTrue(torch.equal(linear(torch.ones(2, 32), weights['plain']), torch.full((2, 4), 32.)))
        self.assertEqual(len(calls), count)
        kernel = SimpleNamespace(linear=lambda x, weight, scales, bias: torch.full((1, 4), 9.))
        with patch.dict('sys.modules', {'damendaemon.quantized_kernels': kernel}):
            self.assertTrue(torch.equal(linear(torch.ones(1, 32), weights['q']), torch.full((1, 4), 9.)))
