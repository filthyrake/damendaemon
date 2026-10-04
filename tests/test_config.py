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


class HybridConfigTests(unittest.TestCase):
    def config(self):
        return {'model_type': 'qwen3_5', 'tie_word_embeddings': True,
                'image_token_id': 99, 'text_config': {
            'model_type': 'qwen3_5_text', 'hidden_size': 1024, 'intermediate_size': 3584,
            'num_hidden_layers': 4, 'num_attention_heads': 8, 'num_key_value_heads': 2,
            'head_dim': 256, 'vocab_size': 248320, 'max_position_embeddings': 262144,
            'linear_num_key_heads': 16, 'linear_num_value_heads': 16,
            'linear_key_head_dim': 128, 'linear_value_head_dim': 128, 'linear_conv_kernel_dim': 4,
            'layer_types': ['linear_attention']*3+['full_attention'], 'hidden_act': 'silu',
            'attn_output_gate': True, 'rms_norm_eps': 1.e-6,
            'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000000., 'partial_rotary_factor': .25}}}

    def test_nested_text_backbone(self):
        c = decoder_config(self.config())
        self.assertEqual(c['rotary_dim'], 64)
        self.assertTrue(c['tie_word_embeddings'])
        self.assertEqual(c['multimodal_token_ids'], [99])

    def test_unsupported_hybrid_variants_fail(self):
        for change in ({'num_experts': 4}, {'layer_types': ['linear_attention']},
                       {'linear_num_value_heads': 17}, {'linear_conv_kernel_dim': 0},
                       {'attention_bias': True}, {'attn_output_gate': False},
                       {'rope_parameters': {'rope_type': 'linear'}},
                       {'rope_parameters': {'rope_type': 'default', 'partial_rotary_factor': .1}}):
            c = self.config()
            c['text_config'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                decoder_config(c)
        c = self.config()
        c['quantization_config'] = {'bits': 4}
        with self.assertRaises(ValueError):
            decoder_config(c)
