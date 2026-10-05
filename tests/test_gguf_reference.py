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

    def test_chunked_hf_reference_preserves_window_boundary_and_continuation(self):
        from transformers import AutoModelForCausalLM, MistralConfig
        from scripts.verify_gguf_reference import reference_prefill
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(71)
            config = MistralConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                   max_position_embeddings=128, sliding_window=16)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                                    attn_implementation='eager').eval()
            for length in (15, 16, 17, 63):
                with self.subTest(length=length):
                    tokens = torch.randint(0, 64, (1, length))
                    full = reference_prefill(model, tokens)
                    chunked = reference_prefill(model, tokens, 7)
                    torch.testing.assert_close(chunked.logits[:, -1], full.logits[:, -1], atol=1e-6, rtol=1e-5)
                    token = full.logits[:, -1].argmax(-1).unsqueeze(1)
                    full = model(token, past_key_values=full.past_key_values, use_cache=True)
                    chunked = model(token, past_key_values=chunked.past_key_values, use_cache=True)
                    torch.testing.assert_close(chunked.logits, full.logits, atol=1e-6, rtol=1e-5)


if __name__=='__main__':
    unittest.main()
