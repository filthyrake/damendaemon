import os
import unittest
from unittest.mock import Mock, patch

import torch

from dllm.kquant_cache import plan
from dllm.quantization import Linear


class KquantCacheTests(unittest.TestCase):
    config = dict(model_type='llama', hidden_size=2048, intermediate_size=5632,
                  num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4,
                  head_dim=64, vocab_size=32000, max_position_embeddings=2048)

    def setUp(self):
        a, b = torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8)
        self.weights = {'embed': a, 'head': a, 'out': b, 'alias': b}
        self.layouts = {'embed': (2048, ((0, 16, 12, 0),)),
                        'head': (2048, ((0, 16, 12, 0),)),
                        'out': (2048, ((0, 16, 14, 0),)),
                        'alias': (2048, ((0, 16, 14, 0),))}
        self.args = dict(config=self.config, weights=self.weights, layouts=self.layouts,
                         device='cuda', dtype=torch.float16, precision='mixed', context=2048,
                         available_bytes=10**9)

    def test_default_off_and_budget_includes_unique_nonembedding_copies(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(plan(**self.args).bytes, 0)
        with patch.dict(os.environ, {'DLLM_KQUANT_DENSE_CACHE_MIB': '1'}):
            chosen = plan(**self.args)
            self.assertEqual(chosen.names, ('head', 'out'))
            self.assertEqual(chosen.bytes, 2 * 2 * 2048 * 16)
            self.assertEqual(plan(**{**self.args, 'available_bytes': chosen.bytes}).bytes, chosen.bytes)
            self.assertEqual(plan(**{**self.args, 'available_bytes': chosen.bytes - 1}).bytes, 0)
            large = {name: (width, ((0, 1024, spec[0][2], 0),))
                     for name, (width, spec) in self.layouts.items()}
            self.assertEqual(plan(**{**self.args, 'layouts': large}).bytes, 0)

    def test_rejects_malformed_budget_and_unqualified_scopes(self):
        for value in ('-1', '1.5', 'nan', 'unlimited'):
            with self.subTest(value=value), patch.dict(os.environ, {'DLLM_KQUANT_DENSE_CACHE_MIB': value}):
                with self.assertRaises(ValueError):
                    plan(**self.args)
        changes = ({'device': 'mps'}, {'dtype': torch.float32}, {'precision': 'float32'},
                   {'context': 0}, {'context': 2049}, {'layouts': {}},
                   {'config': {**self.config, 'num_hidden_layers': 16}},
                   {'config': {**self.config, 'rope_scaling': {'type': 'linear', 'factor': 2}}},
                   {'config': {**self.config, 'sliding_window': 512}},
                   {'layouts': {'head': (2048, ((0, 16, 13, 0),))}})
        with patch.dict(os.environ, {'DLLM_KQUANT_DENSE_CACHE_MIB': '1'}):
            for change in changes:
                with self.subTest(change=change):
                    self.assertEqual(plan(**{**self.args, **change}).bytes, 0)

    def test_multirow_cache_preserves_single_row_decode_and_bias(self):
        packed = self.weights['head']
        dense = torch.ones(16, 2048, dtype=torch.float16)
        decoded = torch.ones(1, 16, dtype=torch.float16)
        decode = Mock(return_value=decoded)
        layer = Linear({'head': packed}, {}, 'cpu', torch.nn.functional.linear,
                       {'head': self.layouts['head']}, packed_decode=decode,
                       packed_cache={id(packed): dense})
        self.assertIs(layer(torch.ones(1, 2048, dtype=torch.float16), packed), decoded)
        decode.reset_mock()
        output = layer(torch.ones(2, 3, 2048, dtype=torch.float16), packed,
                       torch.full((16,), 2, dtype=torch.float16))
        self.assertTrue(torch.equal(output, torch.full((2, 3, 16), 2050, dtype=torch.float16)))
        decode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
