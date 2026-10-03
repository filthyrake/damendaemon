import unittest
from dllm.config import decoder_config


class ConfigTests(unittest.TestCase):
    def config(self, **changes):
        return {'model_type': 'qwen3', 'hidden_size': 1024, 'intermediate_size': 3072,
                'num_attention_heads': 16, 'num_key_value_heads': 8, 'head_dim': 128,
                'num_hidden_layers': 28, 'vocab_size': 151936, 'max_position_embeddings': 40960,
                'hidden_act': 'silu', 'rms_norm_eps': 1e-6, **changes}

    def test_explicit_head_dimension_is_preserved(self):
        self.assertEqual(decoder_config(self.config())['head_dim'], 128)

    def test_implicit_head_dimension_is_derived(self):
        config = self.config(model_type='llama')
        del config['head_dim']
        self.assertEqual(decoder_config(config)['head_dim'], 64)

    def test_unsupported_variants_fail_before_weight_loading(self):
        for changes in ({'model_type': 'qwen3_5'}, {'model_type': 'qwen3_moe'},
                        {'rope_scaling': {'type': 'linear', 'factor': 2}},
                        {'partial_rotary_factor': .25}, {'mlp_bias': True},
                        {'attention_bias': True}, {'layer_types': ['linear_attention']},
                        {'use_sliding_window': True}, {'num_key_value_heads': 3},
                        {'head_dim': 63}, {'quantization_config': {'bits': 4}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decoder_config(self.config(**changes))
