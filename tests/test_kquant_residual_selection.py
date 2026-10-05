import os
import unittest
from unittest.mock import patch

import torch

from dllm.native_kquant_residual import qualified_scope, select
from dllm.quantization import Linear


class ResidualSelectionTests(unittest.TestCase):
    config = {'model_type': 'llama', 'hidden_size': 2048}
    layouts = {'qkv': (2048, ((0, 8, 12, 0), (8, 16, 14, 9216)))}

    def test_only_qualified_model_precision_and_formats(self):
        self.assertTrue(qualified_scope(self.config, self.layouts, 'cuda', torch.float16, 'mixed'))
        for config, layouts, device, dtype in (
            ({**self.config, 'model_type': 'qwen3'}, self.layouts, 'cuda', torch.float16),
            ({**self.config, 'hidden_size': 4096}, self.layouts, 'cuda', torch.float16),
            (self.config, {}, 'cuda', torch.float16),
            (self.config, self.layouts, 'mps', torch.float16),
            (self.config, self.layouts, 'cuda', torch.float32),
            (self.config, {**self.layouts, 'q5': (2048, ((0, 8, 13, 0),))}, 'cuda', torch.float16),
            (self.config, {**self.layouts, 'float': (2048, ((0, 8, 1, 0),))}, 'cuda', torch.float16),
        ):
            with self.subTest(config=config, layouts=layouts, device=device, dtype=dtype):
                self.assertFalse(qualified_scope(config, layouts, device, dtype, 'mixed'))

    def test_standard_precision_is_not_qualified(self):
        self.assertFalse(qualified_scope(self.config, self.layouts, 'cuda', torch.float16, 'standard'))

    def test_opt_in_captures_library_and_falls_back_for_other_shapes(self):
        with patch.dict(os.environ, {}, clear=True), patch('dllm.native_kquant_residual.library') as load:
            self.assertIsNone(select(self.config, self.layouts, 'cuda', torch.float16, 'mixed'))
            load.assert_not_called()
        with patch.dict(os.environ, {'DLLM_KQUANT_RESIDUAL_LIBRARY': '/first.so'}), \
                patch('dllm.native_kquant_residual.library') as load:
            decode = select(self.config, self.layouts, 'cuda', torch.float16, 'mixed')
            load.assert_called_once_with('/first.so')
        weight = torch.zeros(1, dtype=torch.uint8)
        result = torch.ones(1, 16)
        with patch.dict(os.environ, {'DLLM_KQUANT_RESIDUAL_LIBRARY': '/second.so'}), \
                patch('dllm.native_kquant_residual.linear', return_value=result) as multiply:
            self.assertIsNone(decode(torch.zeros(2, 2048), weight, self.layouts['qkv']))
            self.assertIsNone(decode(torch.zeros(1, 8192), weight, (8192, self.layouts['qkv'][1])))
            self.assertIs(decode(torch.zeros(1, 2048), weight, self.layouts['qkv']), result)
            self.assertEqual(multiply.call_args.kwargs['library_path'], '/first.so')
            self.assertEqual(multiply.call_count, 1)

    def test_dispatch_is_per_executor_and_does_not_change_dense_matrices(self):
        packed = torch.zeros(1, dtype=torch.uint8)
        dense = torch.ones(2, 3)
        expected = torch.ones(1, 16)
        def decode(*args):
            return expected
        a = Linear({'packed': packed}, {}, 'cpu', torch.nn.functional.linear,
                   {'packed': self.layouts['qkv']}, decode)
        b = Linear({'packed': packed}, {}, 'cpu', torch.nn.functional.linear,
                   {'packed': self.layouts['qkv']})
        self.assertIs(a(torch.zeros(1, 2048), packed), expected)
        self.assertIsNone(b.packed_decode)
        self.assertTrue(torch.equal(a(torch.ones(1, 3), dense), torch.full((1, 2), 3.)))


if __name__ == '__main__':
    unittest.main()
