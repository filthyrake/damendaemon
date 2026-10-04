import unittest
from dllm.config import rotary_scaling, eos_ids
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

    def test_mistral_window_validation(self):
        for window in (None, 4096):
            config = self.config(model_type='mistral', sliding_window=window)
            self.assertEqual(decoder_config(config)['sliding_window'], window)
        for window in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                decoder_config(self.config(model_type='mistral', sliding_window=window))

    def test_unsupported_variants_fail_before_weight_loading(self):
        for changes in ({'model_type': 'qwen3_5'}, {'model_type': 'qwen3_moe'},
                        {'rope_scaling': {'type': 'yarn', 'factor': 2}},
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


class RotaryConfigTests(unittest.TestCase):
    def test_static_schedules_and_eos_lists(self):
        self.assertEqual(rotary_scaling({'type':'linear','factor':2})['rope_type'],'linear')
        self.assertEqual(eos_ids([1,2,1]),{1,2})
        self.assertEqual(eos_ids(3),{3})
        self.assertEqual(eos_ids(None),set())

    def test_invalid_schedules_fail_before_loading(self):
        base={'rope_type':'llama3','factor':32,'low_freq_factor':1,'high_freq_factor':4,
              'original_max_position_embeddings':8192}
        self.assertEqual(rotary_scaling(base),base)
        for change in ({'factor':float('nan')},{'factor':True},{'factor':.5},
                       {'high_freq_factor':1},{'original_max_position_embeddings':0},
                       {'type':'linear'},{'attention_factor':2}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                rotary_scaling({**base,**change})
        for value in ([],[True],[-1],'2'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                eos_ids(value)


class GemmaConfigTests(unittest.TestCase):
    def config(self, **changes):
        return {**ConfigTests().config(), 'model_type':'gemma3_text',
                'hidden_activation':'gelu_pytorch_tanh', 'sliding_window':512,
                'rope_local_base_freq':10000, 'query_pre_attn_scalar':256,
                'layer_types':['sliding_attention']*27+['full_attention'], **changes}

    def test_preserves_local_attention_and_tied_default(self):
        c = decoder_config(self.config())
        self.assertEqual(c['layer_types'][0], 'sliding_attention')
        self.assertTrue(c['tie_word_embeddings'])

    def test_unsupported_variants_fail_before_loading(self):
        for change in ({'layer_types':[]}, {'sliding_window':0}, {'sliding_window':True},
                       {'query_pre_attn_scalar':float('nan')}, {'rope_local_base_freq':-1},
                       {'use_bidirectional_attention':True}, {'attn_logit_softcapping':50},
                       {'final_logit_softcapping':30}, {'attention_bias':True},
                       {'hidden_activation':'silu'}, {'quantization_config':{'bits':4}},
                       {'rope_scaling':{'type':'linear','factor':2}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                decoder_config(self.config(**change))


class Phi3ConfigTests(unittest.TestCase):
    def test_packed_decoder_and_non_power_of_two_heads(self):
        c = ConfigTests().config(model_type='phi3', hidden_size=3072, num_attention_heads=32,
                                 num_key_value_heads=32, head_dim=96, sliding_window=2047)
        self.assertEqual(decoder_config(c)['head_dim'], 96)
        self.assertEqual(decoder_config(c)['sliding_window'], 2047)
        for window in (False, -1, 0, 3.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                decoder_config({**c, 'sliding_window':window})
