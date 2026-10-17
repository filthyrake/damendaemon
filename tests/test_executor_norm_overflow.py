"""Legacy executor normalizations must fail on overflow instead of silently zeroing.

A finite FP32 input whose square overflows produces an infinite variance; the
rsqrt is then zero and every executor previously returned an all-zero result as
if it were valid. Kernel/fused backends are qualified separately and untouched.
"""
import unittest

import torch
from types import SimpleNamespace

from damendaemon.t5 import T5Executor
from damendaemon.mamba import MambaExecutor
from damendaemon.gemma import GemmaExecutor
from damendaemon.hybrid import HybridExecutor
from damendaemon.qwen2 import Executor as Qwen2Executor


def overflowing(rows=1, width=8):
    value = torch.zeros(rows, width)
    value[..., 0] = 1e20
    return value


class ExecutorNormOverflowTests(unittest.TestCase):
    def test_each_eager_norm_rejects_overflowing_variance(self):
        ones = torch.ones(8)
        cases = {
            't5': (T5Executor.norm, SimpleNamespace(kernels=None, eps=1e-6)),
            'mamba': (MambaExecutor.norm, SimpleNamespace(kernels=None, c={'layer_norm_epsilon': 1e-5})),
            'gemma': (GemmaExecutor.norm, SimpleNamespace(kernels=None, eps=1e-6, compute_dtype=torch.float32)),
            'hybrid': (HybridExecutor.norm, SimpleNamespace(fused=None, eps=1e-5, compute_dtype=torch.float32)),
            'qwen2': (Qwen2Executor.norm, SimpleNamespace(kernels=None, eps=1e-6, precise=False)),
        }
        for name, (norm, fake) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                    norm(fake, overflowing(), ones)

    def test_qwen3_head_norm_rejects_overflowing_heads(self):
        fake = SimpleNamespace(c={'model_type': 'qwen3'}, dim=2, eps=1e-6,
                               w={'0.q_norm': torch.ones(2), '0.k_norm': torch.ones(2)})
        q = overflowing(1, 2).reshape(1, 1, 2)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite QK'):
            Qwen2Executor.normalize_qk(fake, q, q, '0')

    def test_norm_preserves_activation_dtype_for_half_precision(self):
        # The FP32 variance check must not promote FP16/BF16 activations to FP32:
        # the executors declare a residual dtype and downstream linear calls
        # require matching dtypes. T5/Mamba currently pin FP32, so this pins the
        # contract for when half precision is enabled.
        half = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        weight = torch.ones(2, dtype=torch.float16)
        for name, fake in (('t5', SimpleNamespace(kernels=None, eps=1e-6)),
                           ('mamba', SimpleNamespace(kernels=None, c={'layer_norm_epsilon': 1e-5}))):
            norm = {'t5': T5Executor.norm, 'mamba': MambaExecutor.norm}[name]
            out = norm(fake, half, weight)
            self.assertEqual(out.dtype, torch.float16, name)
            self.assertTrue(torch.isfinite(out).all(), name)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            T5Executor.norm(SimpleNamespace(kernels=None, eps=1e-6),
                            overflowing().half(), weight)

    def test_ordinary_inputs_are_unchanged(self):
        value = torch.tensor([[3.0, 4.0]])
        ones = torch.ones(2)
        t5 = T5Executor.norm(SimpleNamespace(kernels=None, eps=1e-6), value, ones)
        torch.testing.assert_close(t5, value * (12.5 + 1e-6) ** -.5)
        torch.testing.assert_close(
            MambaExecutor.norm(SimpleNamespace(kernels=None, c={'layer_norm_epsilon': 1e-5}), value, ones),
            value * (12.5 + 1e-5) ** -.5)
        torch.testing.assert_close(
            GemmaExecutor.norm(SimpleNamespace(kernels=None, eps=1e-6, compute_dtype=torch.float32), value, ones),
            value * (12.5 + 1e-6) ** -.5)
        torch.testing.assert_close(
            HybridExecutor.norm(SimpleNamespace(fused=None, eps=1e-5, compute_dtype=torch.float32), value, ones),
            value * (12.5 + 1e-5) ** -.5)
        torch.testing.assert_close(
            Qwen2Executor.norm(SimpleNamespace(kernels=None, eps=1e-6, precise=False), value, ones),
            value * (12.5 + 1e-6) ** -.5)


if __name__ == '__main__':
    unittest.main()
