import os
import unittest
from unittest.mock import patch

import torch

from dllm.native_kquant_residual_f32 import qualified_scope, select


class Float32ResidualSelectionTests(unittest.TestCase):
    config = {'model_type': 'llama', 'hidden_size': 2048}
    layouts = {'qkv': (2048, ((0, 8, 12, 0), (8, 16, 14, 9216)))}

    def test_qualified_scope_keeps_precision_family_and_context_boundaries(self):
        defaults = dict(config=self.config, layouts=self.layouts, device='cuda',
                        dtype=torch.float32, precision='float32', context=2048)
        self.assertTrue(qualified_scope(**defaults))
        q5 = {'qkv': (2048, ((0, 8, 13, 0), (8, 16, 14, 11264)))}
        self.assertTrue(qualified_scope(**{**defaults, 'layouts': q5}))
        for changed in (
            {'context': 2049}, {'context': 0}, {'precision': 'mixed'},
            {'dtype': torch.float16}, {'device': 'mps'}, {'layouts': {}},
            {'config': {**self.config, 'model_type': 'qwen3'}},
            {'config': {**self.config, 'hidden_size': 4096}},
            {'layouts': {'legacy': (2048, ((0, 8, 2, 0),))}},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(qualified_scope(**{**defaults, **changed}))

    def test_opt_in_is_per_executor_and_preserves_fallback(self):
        args = (self.config, self.layouts, 'cuda', torch.float32, 'float32', 2048)
        with patch.dict(os.environ, {}, clear=True), patch('dllm.native_kquant_residual_f32.library') as load:
            self.assertIsNone(select(*args))
            load.assert_not_called()
        with patch.dict(os.environ, {'DLLM_KQUANT_RESIDUAL_F32_LIBRARY': '/first.so'}), \
                patch('dllm.native_kquant_residual_f32.library') as load:
            decode = select(*args)
            load.assert_called_once_with('/first.so')
        weight = torch.zeros(1, dtype=torch.uint8)
        result = torch.ones(1, 16)
        with patch.dict(os.environ, {'DLLM_KQUANT_RESIDUAL_F32_LIBRARY': '/second.so'}), \
                patch('dllm.native_kquant_residual_f32.linear', return_value=result) as multiply:
            for rows in (1, 2, 4):
                for width in (2048, 8192):
                    self.assertIs(decode(torch.zeros(rows, width), weight,
                                         (width, self.layouts['qkv'][1])), result)
                    self.assertEqual(multiply.call_args.kwargs['library_path'], '/first.so')
            self.assertIsNone(decode(torch.zeros(5, 2048), weight, self.layouts['qkv']))
            self.assertIsNone(decode(torch.zeros(1, 4096), weight, (4096, self.layouts['qkv'][1])))
            self.assertEqual(multiply.call_count, 6)


if __name__ == '__main__':
    unittest.main()
