"""Hybrid executor per-mode KV byte accounting (#535 gate 3); no GPU qualification."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from damendaemon.config import decoder_config
from damendaemon.hybrid import HybridExecutor, load_weights
from tests.test_capacity_cli import fixture
from tests.test_xpu import ENV


class KvStatusAccountingTests(unittest.TestCase):
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

    def test_status_accounting_reports_mode_and_resident_bytes(self):
        baseline = self.executor(None)
        self.assertEqual(baseline.kv_mode, 'none')
        expected = sum(cache.numel()*cache.element_size() for cache in baseline.kv.values())
        self.assertEqual(baseline.kv_cache_bytes, expected)
        quantized = self.executor('q8_0')
        self.assertEqual(quantized.kv_mode, 'q8_0')
        expected = sum(data.numel() + scales.numel()*scales.element_size()
                       for data, scales in quantized.kv.values())
        self.assertEqual(quantized.kv_cache_bytes, expected)
        self.assertLess(quantized.kv_cache_bytes, baseline.kv_cache_bytes/3)
        # Non-hybrid executors leave the fields absent so status falls back to
        # None rather than reporting unrelated executor attributes.


if __name__ == '__main__':
    unittest.main()
