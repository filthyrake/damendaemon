"""Synthetic Linux accounting and real CPU weights; no hardware qualification."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from damendaemon import host_memory
from damendaemon.config import decoder_config
from damendaemon.hybrid import load_weights
from tests.test_capacity_cli import fixture

ENV = {'DAMENDAEMON_DEVICE': 'cuda', 'DAMENDAEMON_KERNELS': 'torch',
       'DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD': '1'}


class HostMemoryTests(unittest.TestCase):
    def test_accounting_fails_closed_and_never_substitutes_swap_or_free(self):
        with patch.object(Path, 'read_text', return_value='MemFree: 1 kB\nMemAvailable: 123 kB\n'):
            self.assertEqual(host_memory.available_memory(), 123 * 1024)
        for text in ('MemFree: 999999 kB\nSwapFree: 999999 kB', 'MemAvailable: -1 kB',
                     'MemAvailable: NaN kB', 'MemAvailable: 100 MB',
                     'MemAvailable: 1 kB\nMemAvailable: 2 kB', 'MemAvailable: 1 kB extra'):
            with self.subTest(text=text), patch.object(Path, 'read_text', return_value=text), \
                    self.assertRaisesRegex(RuntimeError, 'MemAvailable'):
                host_memory.available_memory()
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError), self.assertRaises(RuntimeError):
            host_memory.available_memory()

    def test_pending_allocation_cannot_cross_reserve(self):
        with patch.object(host_memory, 'available_memory', return_value=host_memory.HOST_RESERVE + 1024):
            self.assertEqual(host_memory.require_headroom(1023), 1024)
            for pending in (1024, 1025):
                with self.assertRaises(MemoryError):
                    host_memory.require_headroom(pending)
            for invalid in (-1, True, 1.5):
                with self.assertRaises(ValueError):
                    host_memory.require_headroom(invalid)

    def test_checkpoint_header_budget_and_early_load_rejection(self):
        from damendaemon.qwen2 import ModelWeights
        with TemporaryDirectory() as directory:
            path = Path(directory)
            _, tensors = fixture(path)
            # The pop-based loader never holds source + full converted copy:
            # the budget is the whole source plus the LARGEST single-tensor
            # fp32 conversion transient (the worst conversion direction).
            # The fixture writes F32 sources (4 B/elem).
            expected = (sum(t.numel()*t.element_size() for t in tensors.values())
                        + max(t.numel()*4 for t in tensors.values()))
            with patch.object(host_memory, 'require_headroom') as check:
                self.assertEqual(host_memory.admit_checkpoint([path/'model.safetensors']), expected)
                check.assert_called_once_with(expected)
            # No tokenizer or GPU load is reached when the entire load cannot fit.
            with patch.dict('os.environ', ENV, clear=True), \
                    patch.object(host_memory, 'available_memory', return_value=host_memory.HOST_RESERVE), \
                    patch('damendaemon.qwen2.Tokenizer') as tokenizer, \
                    patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU initialized')), \
                    self.assertRaises(MemoryError):
                ModelWeights(path, 32, 'none')
            tokenizer.from_file.assert_not_called()

    def test_guarded_conversion_preserves_weights_and_checks_concatenation(self):
        with TemporaryDirectory() as directory:
            config, tensors = fixture(Path(directory))
        config = decoder_config(config)
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda'}, clear=True):
            expected = load_weights(dict(tensors), config)
        with patch.dict('os.environ', ENV, clear=True), \
                patch.object(host_memory, 'require_headroom') as check:
            actual = load_weights(dict(tensors), config)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        self.assertIs(actual['head'], actual['embed'])
        # Concatenation has its own pending-byte check after input conversion.
        self.assertIn(unittest.mock.call(actual['0.gate_up'].numel()*2), check.call_args_list)
        with patch.dict('os.environ', ENV, clear=True), \
                patch.object(host_memory, 'require_headroom', side_effect=MemoryError('pressure')), \
                patch('torch.cat') as cat, self.assertRaises(MemoryError):
            load_weights(dict(tensors), config)
        cat.assert_not_called()

    def test_pressure_after_admission_rejects_source_read(self):
        from damendaemon.qwen2 import ModelWeights
        with TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            (path/'tokenizer_config.json').write_text('{}')
            (path/'generation_config.json').write_text('{"eos_token_id": 1}')
            with patch.dict('os.environ', ENV, clear=True), \
                    patch('damendaemon.qwen2.Tokenizer'), \
                    patch.object(host_memory, 'available_memory', side_effect=[2**40, 0]), \
                    patch('damendaemon.hybrid.load_weights') as mapping, \
                    self.assertRaises(MemoryError):
                ModelWeights(path, 32, 'none')
            mapping.assert_not_called()

    def test_offline_meta_planning_never_queries_host(self):
        from damendaemon.capacity import plan_local
        with TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            with patch.dict('os.environ', ENV, clear=True), \
                    patch.object(host_memory, 'available_memory', side_effect=AssertionError('live accounting')):
                report = plan_local(path, [32], [1], 'float32')
            self.assertEqual(report['runtime_qualification'], 'not assessed')
