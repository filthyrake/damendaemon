from types import SimpleNamespace
import unittest
from unittest.mock import patch
from dllm.metal_q4_policy import validate


class MetalQ4PolicyTests(unittest.TestCase):
    def config(self):
        return dict(model_type='qwen2', hidden_size=896, num_hidden_layers=24,
                    num_attention_heads=14, num_key_value_heads=2, intermediate_size=4864,
                    vocab_size=151936, head_dim=64)

    def test_qualified_bounds_and_unqualified_combinations(self):
        validate(self.config(), 2048, 4)
        for context in (0, 2049, True):
            with self.assertRaises(ValueError):validate(self.config(), context, 4)
        for changes in ({'model_type':'qwen3'}, {'hidden_size':1024}, {'rope_scaling':{'type':'linear','factor':2}}):
            with self.assertRaises(ValueError):validate(self.config() | changes, 512, 4)
        for batch in (0, 5, True):
            with self.assertRaises(ValueError):validate(self.config(), 512, batch)

    def test_unqualified_batch_rejected_before_gpu_construction(self):
        from dllm.qwen2 import Executor
        model = SimpleNamespace(config=self.config(), max_context=512, quantization='q4_0')
        with patch('dllm.qwen2.selected_device', return_value='mps'):
            with self.assertRaisesRegex(ValueError, 'request slots'):Executor(model, max_batch=5)
