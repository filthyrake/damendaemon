"""Hybrid Q4 opt-in must not silently broaden vendor, context or image support."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from dllm.hybrid_q4_policy import validate
from dllm.hybrid_precision import select


class HybridQ4Tests(unittest.TestCase):
    env = {'DLLM_DEVICE': 'rocm', 'DLLM_EXPERIMENTAL_HYBRID_Q4': '1',
           'DLLM_HYBRID_PRECISION': 'compensated', 'DLLM_HYBRID_PREFILL': 'scan'}
    config = {'model_type': 'qwen3_5_text'}

    def test_explicit_scope_and_other_precision_guards(self):
        model = SimpleNamespace(config=self.config, max_context=512, quantization='q4_0')
        with patch.dict('os.environ', self.env, clear=True), patch('dllm.hybrid_q4_policy.device_name', return_value='NVIDIA A2'):
            self.assertEqual(select(model, False, 4, 'triton'), 'compensated')
            for graph, batch, kernels in ((True, 4, 'triton'), (False, 5, 'triton'), (False, 4, 'torch')):
                with self.assertRaises(ValueError):
                    select(model, graph, batch, kernels)
            for context in (0, 513):
                with self.assertRaises(ValueError):
                    validate(self.config, context)
            for changes in ({'DLLM_EXPERIMENTAL_HYBRID_Q4': '0'},
                            {'DLLM_DEVICE': 'cuda'}, {'DLLM_DEVICE': 'mps'},
                            {'DLLM_EXPERIMENTAL_VISION': '1'},
                            {'DLLM_HYBRID_PRECISION': 'float16'},
                            {'DLLM_HYBRID_PRECISION': 'float32'}):
                with patch.dict('os.environ', changes):
                    with self.assertRaises(ValueError):
                        select(model, False, 4, 'triton')

    def test_a30_requires_q4_opt_in_and_retains_execution_bounds(self):
        model = SimpleNamespace(config=self.config, max_context=512, quantization='q4_0')
        with patch.dict('os.environ', self.env | {'DLLM_DEVICE': 'cuda'}, clear=True), \
                patch('dllm.hybrid_q4_policy.device_name', return_value='NVIDIA A30'):
            self.assertEqual(select(model, False, 4, 'triton'), 'compensated')
            for changes in ({'max_context': 513}, {'quantization': 'q8_0'}):
                with self.assertRaises(ValueError):
                    select(SimpleNamespace(**(vars(model) | changes)), False, 4, 'triton')
            with self.assertRaises(ValueError):
                select(model, True, 4, 'triton')
            with self.assertRaises(ValueError):
                select(model, False, 5, 'triton')
            with patch.dict('os.environ', {'DLLM_EXPERIMENTAL_HYBRID_Q4': '0'}):
                with self.assertRaises(ValueError):
                    select(model, False, 4, 'triton')

    def test_no_opt_in_and_wrong_architecture_rejected(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                validate(self.config, 512)
        with patch.dict('os.environ', self.env, clear=True):
            with self.assertRaises(ValueError):
                validate({'model_type': 'qwen3'}, 512)

    def test_chunked_prefill_stays_within_admitted_q4_cuda_path(self):
        from dllm.hybrid import HybridExecutor
        executor = HybridExecutor.__new__(HybridExecutor)
        executor.model = SimpleNamespace(quantization='q4_0')
        executor.hybrid_precision = 'compensated'
        executor.fused = object()
        executor.chunk_scan = None
        with patch.dict('os.environ', {'DLLM_DEVICE': 'cuda'}):
            self.assertTrue(executor.supports_chunked_prefill)
            self.assertFalse(executor.requires_full_prefill)
            for target, name, value in ((executor.model, 'quantization', 'q8_0'),
                                        (executor, 'hybrid_precision', 'float16'),
                                        (executor, 'fused', None),
                                        (executor, 'chunk_scan', object())):
                with patch.object(target, name, value):
                    self.assertFalse(executor.supports_chunked_prefill)
        with patch.dict('os.environ', {'DLLM_DEVICE': 'mps'}):
            self.assertFalse(executor.supports_chunked_prefill)

    def test_packed_header_budget_keeps_embedding_and_norm_precision(self):
        from dllm.qwen2 import weight_bytes
        config = self.config | {'tie_word_embeddings': True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text('{}')
            save_file({'model.language_model.embed_tokens.weight': torch.zeros(64, 32),
                       'model.language_model.norm.weight': torch.zeros(32),
                       'model.language_model.layers.0.self_attn.q_proj.weight': torch.zeros(32, 32)},
                      str(path/'model.safetensors'))
            with patch('dllm.qwen2.decoder_config', return_value=config):
                with patch.dict('os.environ', self.env, clear=True):
                    # FP16 embedding + FP32 norm + packed projection + tied packed head.
                    self.assertEqual(weight_bytes(path, 'q4_0'), 4096 + 128 + 576 + 1152)
                with patch.dict('os.environ', {}, clear=True):
                    with self.assertRaises(ValueError):
                        weight_bytes(path, 'q4_0')

    def test_callback_routes_only_q4_and_preserves_scalar_fallback(self):
        from dllm.quantization import Linear
        weights = {'q': torch.zeros((4, 16), dtype=torch.uint8), 'plain': torch.ones((4, 32))}
        scales = {'q': torch.ones((4, 1), dtype=torch.float16)}
        calls = []
        def projection(x, weight, scale, bias):
            calls.append((weight, scale))
            return torch.full((x.shape[0], 4), 7.) if x.shape[0] >= 2 else None
        linear = Linear(weights, scales, 'cpu', torch.nn.functional.linear, q4_projection=projection)
        self.assertTrue(torch.equal(linear(torch.ones(2, 32), weights['q']), torch.full((2, 4), 7.)))
        self.assertIs(calls[0][0], weights['q'])
        self.assertIs(calls[0][1], scales['q'])
        self.assertTrue(torch.equal(linear(torch.ones(2, 32), weights['plain']), torch.full((2, 4), 32.)))
        self.assertEqual(len(calls), 1)
        kernel = SimpleNamespace(linear=lambda x, w, s, b: torch.full((1, 4), 9.))
        with patch.dict('sys.modules', {'dllm.quantized_kernels_q4': kernel}):
            self.assertTrue(torch.equal(linear(torch.ones(1, 32), weights['q']), torch.full((1, 4), 9.)))

    def test_q8_context_probe_cannot_bypass_q4_context_limit(self):
        from scripts.experiments.hybrid_context_probe import context_probe
        model = SimpleNamespace(config=self.config, max_context=4096, quantization='q4_0')
        from dllm import hybrid_precision
        with patch.dict('os.environ', self.env, clear=True), context_probe():
            with self.assertRaises(ValueError):
                hybrid_precision.select(model, False, 4, 'triton')

    def test_packed_output_head_uses_quantized_dispatch(self):
        from dllm.hybrid import HybridExecutor
        for dtype in (torch.int8, torch.uint8):
            executor = HybridExecutor.__new__(HybridExecutor)
            executor.c = {'layer_types': []}
            executor.w = {'head': torch.zeros((8, 16), dtype=dtype), 'norm': torch.ones(32)}
            executor.norm = Mock(side_effect=lambda value, weight: value)
            executor.linear = Mock(return_value=torch.ones(1, 8))
            executor.fused = SimpleNamespace(output_head=Mock(side_effect=AssertionError('packed head entered dense kernel')))
            actual = executor.output_logits(torch.ones(1, 32))
            self.assertEqual(actual.shape, (1, 8))
            self.assertIs(executor.linear.call_args.args[1], executor.w['head'])
            self.assertEqual(executor.linear.call_args.args[0].dtype, torch.float32)
            executor.fused.output_head.assert_not_called()
