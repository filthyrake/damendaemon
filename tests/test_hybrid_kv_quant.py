"""Real CPU tensor regressions for the quantized hybrid KV cache; no GPU qualification."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from damendaemon.config import decoder_config
from damendaemon.hybrid import HybridExecutor, load_weights
from damendaemon.kv_quant import group_for
from tests.test_capacity_cli import fixture
from tests.test_xpu import ENV


class QuantizedKvTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict('os.environ', ENV, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with TemporaryDirectory() as directory:
            config, tensors = fixture(Path(directory))
        self.config = decoder_config(config)
        rng = torch.Generator().manual_seed(321)
        tensors = {name: torch.randn(t.shape, generator=rng)*.08 for name, t in tensors.items()}
        self.weights = load_weights(tensors, self.config)

    def executor(self, kv_quant):
        weights = {name: value.float() for name, value in self.weights.items()}
        weights['head'] = weights['embed']
        model = SimpleNamespace(config=self.config, max_context=32, weights=weights, scales={},
            quantization='none', bytes=sum(t.numel()*t.element_size() for t in
                {id(t):t for t in weights.values()}.values()))
        environment = {'DAMENDAEMON_HYBRID_KV': kv_quant} if kv_quant else {}
        with patch.dict('os.environ', environment), \
                patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')

    def test_executor_builds_packed_cache_and_keeps_fp32_default(self):
        quantized = self.executor('q8_0')
        for cache in quantized.kv.values():
            data, scales = cache
            self.assertEqual(data.dtype, torch.int8)
            self.assertEqual(scales.dtype, torch.float16)
            group = group_for(data.shape[-1])
            self.assertEqual(data.shape[-1] // group, scales.shape[-1])
            # Resident pair is packed: int8 values plus FP16 scales are far
            # smaller than the FP32 cache the baseline allocates.
        baseline = self.executor(None)
        for cache in baseline.kv.values():
            self.assertEqual(cache.dtype, torch.float32)
        quantized_bytes = sum(d.numel() + s.numel()*2 for d, s in quantized.kv.values())
        baseline_bytes = sum(cache.numel()*cache.element_size() for cache in baseline.kv.values())
        self.assertLess(quantized_bytes, baseline_bytes/3)

    def test_q8_kv_batched_decode_with_divergent_histories(self):
        # Two slots prefill different prompts, then decode jointly: rows carry
        # different absolute positions in one forward pass.
        fp32 = self.executor(None)
        q8 = self.executor('q8_0')
        prompts = ([1, 4, 2, 7], [7, 5, 3])
        fp32.prefill_batch(prompts, [0, 1])
        q8.prefill_batch(prompts, [0, 1])
        for _ in range(3):
            _, reference_next = fp32.decode_batch([9, 8], [0, 1])
            _, actual_next = q8.decode_batch([9, 8], [0, 1])
            self.assertEqual(actual_next[0].item(), reference_next[0].item())
            self.assertEqual(actual_next[1].item(), reference_next[1].item())

    def test_invalid_kv_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DAMENDAEMON_HYBRID_KV'):
            self.executor('q4_0')

    def test_q8_kv_prefill_and_decode_track_fp32_within_quantization_noise(self):
        tokens = [1, 4, 2, 7, 5, 3, 8, 6, 9]
        fp32 = self.executor(None)
        q8 = self.executor('q8_0')
        reference = fp32.prefill(tokens)
        actual = q8.prefill(tokens)
        torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.05)
        # Each executor follows its own greedy trajectory from its own prefill.
        next_fp32, next_q8 = int(reference[0].argmax(-1)), int(actual[0].argmax(-1))
        for _ in range(4):
            self.assertEqual(next_q8, next_fp32)
            _, next_fp32 = fp32.decode(next_fp32)
            _, next_q8 = q8.decode(next_q8)
        self.assertEqual(next_q8, next_fp32)

    def test_q8_kv_slots_isolate_and_match_fp32_decode(self):
        fp32 = self.executor(None)
        q8 = self.executor('q8_0')
        prompts = ([1, 4, 2], [7, 5, 3])
        fp32.prefill_batch(prompts, [0, 1])
        q8.prefill_batch(prompts, [0, 1])
        for slot in (0, 1):
            reference_logits, reference_next = fp32.decode_batch([9], [slot])
            actual_logits, actual_next = q8.decode_batch([9], [slot])
            torch.testing.assert_close(actual_logits, reference_logits, rtol=0.05, atol=0.05)
            self.assertEqual(int(actual_next), int(reference_next))
        # Slot isolation: the quantized cache keeps per-slot storage.
        for cache in q8.kv.values():
            data, scales = cache
            self.assertFalse(torch.equal(data[0, 0], data[0, 1]))


if __name__ == '__main__':
    unittest.main()
