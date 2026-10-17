import os
import unittest
from unittest.mock import patch

import torch

from damendaemon.kquant_prefill import select
from damendaemon.quantization import Linear


class PrefillSelectionTests(unittest.TestCase):
    config = {'model_type': 'llama', 'hidden_size': 2048}
    layouts = {'qkv': (2048, ((0, 8, 12, 0), (8, 16, 14, 9216)))}

    def test_opt_in_and_qualified_scope(self):
        args = dict(config=self.config, layouts=self.layouts, device='cuda',
                    dtype=torch.float32, precision='float32', context=2048)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(select(**args))
        with patch.dict(os.environ, {'DAMENDAEMON_KQUANT_PREFILL_F32': '1'}):
            self.assertIsNotNone(select(**args))
            self.assertIsNotNone(select(**{**args, 'layouts': {'q5': (2048, ((0, 8, 13, 0), (8, 16, 14, 11264)))}}))
            for changed in ({'context': 2049}, {'dtype': torch.float16}, {'precision': 'mixed'},
                            {'device': 'mps'}, {'layouts': {}},
                            {'config': {**self.config, 'model_type': 'qwen3'}},
                            {'config': {**self.config, 'hidden_size': 4096}},
                            {'layouts': {'legacy': (2048, ((0, 8, 2, 0),))}}):
                with self.subTest(changed=changed):
                    self.assertIsNone(select(**{**args, **changed}))

    def test_prefill_shape_bias_and_executor_isolation(self):
        with patch.dict(os.environ, {'DAMENDAEMON_KQUANT_PREFILL_F32': '1'}):
            selected = select(self.config, self.layouts, 'cuda', torch.float32, 'float32', 512)
        weight = torch.zeros(1, dtype=torch.uint8)
        with patch.dict(os.environ, {}, clear=True), patch('damendaemon.kquant_prefill.linear') as multiply:
            self.assertIsNone(selected(torch.zeros(127, 2048), weight, self.layouts['qkv']))
            self.assertIsNone(selected(torch.zeros(128, 4096), weight, (4096, self.layouts['qkv'][1])))
            multiply.assert_not_called()
            multiply.return_value = torch.ones(128, 16)
            output = selected(torch.zeros(2, 64, 2048), weight, self.layouts['qkv'], torch.ones(16))
            self.assertEqual(tuple(output.shape), (2, 64, 16))
            self.assertTrue(torch.equal(output, torch.full((2, 64, 16), 2.)))
            self.assertEqual(tuple(multiply.call_args.args[0].shape), (128, 2048))

    def test_linear_keeps_decode_prefill_and_dense_paths_separate(self):
        packed = torch.zeros(1, dtype=torch.uint8)
        dense = torch.ones(16, 2048)
        decode_result = torch.ones(1, 16)
        prefill_result = torch.ones(128, 16)
        decode = lambda x, *args: decode_result if x.shape[0] == 1 else None
        prefill = lambda x, *args: prefill_result
        layer = Linear({'packed': packed}, {}, 'cpu', torch.nn.functional.linear,
                       {'packed': self.layouts['qkv']}, decode, prefill)
        self.assertIs(layer(torch.zeros(1, 2048), packed), decode_result)
        self.assertIs(layer(torch.zeros(128, 2048), packed), prefill_result)
        self.assertTrue(torch.equal(layer(torch.ones(1, 2048), dense), torch.full((1, 16), 2048.)))


if __name__ == '__main__':
    unittest.main()
