"""Real CPU tensor regressions for shared eager state; no GPU qualification."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from damendaemon.capacity import hybrid_state_bytes
from damendaemon.config import decoder_config
from damendaemon.hybrid import HybridExecutor, load_weights
from tests.test_capacity_cli import fixture
from tests.test_xpu import ENV


class HybridEagerChunkTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        with TemporaryDirectory() as directory:
            config, tensors = fixture(Path(directory))
        self.config = decoder_config(config)
        rng = torch.Generator().manual_seed(123)
        tensors = {name: torch.randn(t.shape, generator=rng)*.08 for name, t in tensors.items()}
        self.weights = load_weights(tensors, self.config)

    def model(self, fp32=True):
        weights = {name: value.float() for name, value in self.weights.items()} if fp32 else self.weights
        # Preserve the loader's tied embedding/head identity after dtype conversion.
        weights['head'] = weights['embed']
        return SimpleNamespace(config=self.config, max_context=32, weights=weights, scales={},
            quantization='none', bytes=sum(t.numel()*t.element_size() for t in
                {id(t):t for t in weights.values()}.values()))

    def executor(self):
        # Execute actual shared tensor arithmetic on CPU. Device API routing has
        # separate tests; this is deliberately not presented as XPU execution.
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(self.model(), use_graph=False, max_batch=2, kernels='torch')

    def assert_state(self, left, right, slot=0):
        for name in ('state', 'conv'):
            for layer in getattr(left, name):
                torch.testing.assert_close(getattr(left, name)[layer][slot], getattr(right, name)[layer][slot],
                                           rtol=2e-5, atol=2e-6)
        for layer in left.kv:
            length = left.lengths[slot]
            torch.testing.assert_close(left.kv[layer][:, slot, :, :length], right.kv[layer][:, slot, :, :length],
                                       rtol=2e-5, atol=2e-6)

    def test_chunks_preserve_recurrent_convolution_kv_and_decode_state(self):
        tokens = [1, 4, 2, 7, 5, 3, 8, 6, 9]
        for sizes in ((1, 1, 2, 5), (3, 3, 3), (4, 5)):
            with self.subTest(sizes=sizes):
                full, chunked = self.executor(), self.executor()
                target = full.prefill(tokens)
                offset = 0
                for size in sizes:
                    final = offset+size == len(tokens)
                    actual = chunked.prefill_chunk(tokens[offset:offset+size], 0, offset, final)
                    if not final:
                        self.assertIsNone(actual)
                    offset += size
                torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
                self.assert_state(full, chunked)
                for token in (10, 11, 12):
                    torch.testing.assert_close(chunked.decode(token), full.decode(token), rtol=2e-5, atol=2e-6)
                    self.assert_state(full, chunked)

    def test_interleaved_slots_and_reused_slot_reset(self):
        active, control = self.executor(), self.executor()
        a, b = [1, 2, 3, 4, 5], [6, 7, 8]
        active.prefill_chunk(a[:2], 0, 0, False)
        active.prefill(b, slot=1)
        untouched = {name: {i: tensor[1].clone() for i, tensor in getattr(active, name).items()}
                     for name in ('state', 'conv')}
        output = active.prefill_chunk(a[2:], 0, 2)
        torch.testing.assert_close(output, control.prefill(a), rtol=2e-5, atol=2e-6)
        for name, layers in untouched.items():
            for layer, tensor in layers.items():
                torch.testing.assert_close(getattr(active, name)[layer][1], tensor, rtol=0, atol=0)
        active.prefill_chunk(b[:1], 0, 0, False)
        output = active.prefill_chunk(b[1:], 0, 1)
        torch.testing.assert_close(output, control.prefill(b), rtol=2e-5, atol=2e-6)
        self.assert_state(active, control)

    def test_failure_invalidates_partial_slot_and_allows_new_prompt(self):
        executor = self.executor()
        executor.prefill_chunk([1, 2], 0, 0, False)
        with patch.object(executor, 'forward', side_effect=RuntimeError('injected layer failure')):
            with self.assertRaises(RuntimeError):
                executor.prefill_chunk([3], 0, 2)
        self.assertEqual(executor.lengths[0], 0)
        with self.assertRaises(ValueError):
            executor.prefill_chunk([3], 0, 2)
        torch.testing.assert_close(executor.prefill_chunk([4, 5], 0, 0), self.executor().prefill([4, 5]))

    def test_xpu_fp32_storage_budget_precedes_transfer(self):
        model = self.model(fp32=False)
        weights = sum(t.numel()*4 for t in {id(t):t for t in model.weights.values()}.values())
        expected = weights + hybrid_state_bytes(self.config, 32, 2, 4) + 512*1024**2
        executor = HybridExecutor.__new__(HybridExecutor)
        with patch('damendaemon.hybrid.empty_cache'), patch('damendaemon.hybrid.available_memory', return_value=expected-1):
            with self.assertRaises(MemoryError):
                executor.__init__(model, use_graph=False, max_batch=2, kernels='torch')
        self.assertEqual(executor.required_memory_bytes, expected)
        self.assertEqual(executor.hybrid_precision, 'float32')
        self.assertIsNone(executor.fused)
