import os
import unittest
from unittest.mock import patch

import torch

from dllm.native_kquant_residual import qualified_scope, select
from dllm.quantization import Linear


class ResidualSelectionTests(unittest.TestCase):
    config = {'model_type': 'llama', 'hidden_size': 2048}
    layouts = {'qkv': (2048, ((0, 8, 12, 0), (8, 16, 14, 9216)))}

    def test_tinyllama_long_context_does_not_expand_other_model_scopes(self):
        config=dict(self.config,intermediate_size=5632,num_hidden_layers=22,
                    num_attention_heads=32,num_key_value_heads=4,head_dim=64,
                    vocab_size=32000,max_position_embeddings=2048,rope_theta=10000)
        def accepted(c=config,context=2048,layouts=self.layouts):
            return qualified_scope(c,layouts,'cuda',torch.float16,'mixed',context)
        self.assertTrue(accepted())
        for changes in ({'intermediate_size':8192},{'num_hidden_layers':16},
                        {'num_attention_heads':16},{'num_key_value_heads':8},
                        {'head_dim':128},{'vocab_size':128256},
                        {'max_position_embeddings':4096},{'rope_theta':500000},
                        {'rope_scaling':{'type':'linear','factor':2}}, {'sliding_window':1024}):
            with self.subTest(changes=changes):
                self.assertFalse(accepted({**config,**changes}))
        self.assertFalse(accepted(context=2049))
        self.assertFalse(accepted(context=0))
        self.assertFalse(accepted(layouts={'q5':(2048,((0,8,13,0),))}))

    def test_only_qualified_model_precision_and_formats(self):
        self.assertTrue(qualified_scope(self.config, self.layouts, 'cuda', torch.float16, 'mixed', 512))
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
                self.assertFalse(qualified_scope(config, layouts, device, dtype, 'mixed', 512))

    def test_standard_precision_is_not_qualified(self):
        self.assertFalse(qualified_scope(self.config, self.layouts, 'cuda', torch.float16, 'standard', 512))

    def test_long_context_is_not_qualified(self):
        self.assertFalse(qualified_scope(self.config, self.layouts, 'cuda', torch.float16, 'mixed', 2048))

    def test_opt_in_captures_library_and_falls_back_for_other_shapes(self):
        with patch.dict(os.environ, {}, clear=True), patch('dllm.native_kquant_residual.library') as load:
            self.assertIsNone(select(self.config, self.layouts, 'cuda', torch.float16, 'mixed', 512))
            load.assert_not_called()
        with patch.dict(os.environ, {'DLLM_KQUANT_RESIDUAL_LIBRARY': '/first.so'}), \
                patch('dllm.native_kquant_residual.library') as load:
            decode = select(self.config, self.layouts, 'cuda', torch.float16, 'mixed', 512)
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
