import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from test_gguf import fixture, HAS_GGUF


@unittest.skipUnless(HAS_GGUF and importlib.util.find_spec('transformers'), 'optional reference dependencies required')
class GgufReferencePolicyTests(unittest.TestCase):
    def test_fp32_kquant_head_keeps_unrounded_values_with_tied_embedding(self):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        from scripts.verify_gguf_reference import reference_weights
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.gguf'
            fixture(path,kind=12,hidden=256,packed_embedding=True,tied=True,mixed=True)
            tensors={t.name:torch.from_numpy(dequantize(t.data,t.tensor_type).copy())
                     for t in GGUFReader(path).tensors}
            ordinary,_,_=reference_weights(path)
            precise,_,_=reference_weights(path,'kquant-float32')
            self.assertIs(ordinary['lm_head.weight'],ordinary['model.embed_tokens.weight'])
            self.assertTrue(torch.equal(precise['model.embed_tokens.weight'],tensors['token_embd.weight'].half()))
            self.assertEqual(precise['model.embed_tokens.weight'].dtype,torch.float16)
            self.assertEqual(precise['lm_head.weight'].dtype,torch.float32)
            self.assertTrue(torch.equal(precise['lm_head.weight'],tensors['token_embd.weight']))
            self.assertFalse(torch.equal(precise['lm_head.weight'],precise['model.embed_tokens.weight'].float()))
            for axis in ('q','k','v'):
                self.assertTrue(torch.equal(precise[f'model.layers.0.self_attn.{axis}_proj.weight'],
                                            tensors[f'blk.0.attn_{axis}.weight']))
            self.assertTrue(torch.equal(precise['model.norm.weight'],tensors['output_norm.weight'].half()))

    def test_fp32_kquant_policy_rejects_uncovered_legacy_formats(self):
        from scripts.verify_gguf_reference import reference_weights
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.gguf'
            fixture(path,kind=8)
            with self.assertRaisesRegex(ValueError,'legacy'):
                reference_weights(path,'kquant-float32')


if __name__=='__main__':
    unittest.main()
