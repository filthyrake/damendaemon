import os
import unittest
from unittest.mock import patch
import torch
from dllm.native_awq import qualified_scope, select

CONFIG = dict(model_type='qwen2', hidden_size=896, intermediate_size=4864,
              num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
              head_dim=64, vocab_size=151936)


class NativeAwqSelectionTests(unittest.TestCase):
    def test_unsupported_scope_never_loads_native_code(self):
        cases = [(CONFIG, 'cuda', torch.float16, 2048, 5),
                 (CONFIG, 'cuda', torch.float16, 2048, 0),
                 (CONFIG, 'cuda', torch.float16, 0),
                 (CONFIG, 'cuda', torch.float16, 2049),
                 (CONFIG, 'mps', torch.float16, 2048),
                 (CONFIG, 'cuda', torch.float32, 2048)]
        for key in CONFIG:
            cases.append(({**CONFIG, key: None}, 'cuda', torch.float16, 2048))
        with patch.dict(os.environ, {'DLLM_AWQ_MARLIN_LIBRARY': '/invalid'}), \
             patch('torch.cuda.get_device_name', side_effect=AssertionError('CUDA queried')):
            for args in cases:
                self.assertFalse(qualified_scope(*args))
                self.assertIsNone(select(*args))

    def test_opt_in_and_hardware_gate(self):
        args = (CONFIG, 'cuda', torch.float16, 2048)
        with patch.dict(os.environ, {}, clear=True), \
             patch('torch.cuda.get_device_name', side_effect=AssertionError('CUDA queried')):
            self.assertIsNone(select(*args))
        with patch.dict(os.environ, {'DLLM_AWQ_MARLIN_LIBRARY': '/qualified'}), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA A30'), \
             patch('dllm.native_awq.Library', side_effect=AssertionError('library loaded')):
            self.assertIsNone(select(*args))
        with patch.dict(os.environ, {'DLLM_AWQ_MARLIN_LIBRARY': '/qualified'}), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA A2'), \
             patch('dllm.native_awq.Library', side_effect=RuntimeError('bad ABI')):
            with self.assertRaisesRegex(RuntimeError, 'bad ABI'):
                select(*args)
