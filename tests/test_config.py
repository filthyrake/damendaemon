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

    def test_starcoder_layernorm_and_gelu_configuration(self):
        config = self.config(model_type='starcoder2', hidden_act='gelu_pytorch_tanh',
                             norm_epsilon=1e-5, sliding_window=4096, use_bias=True)
        parsed = decoder_config(config)
        self.assertEqual(parsed['rms_norm_eps'], 1e-5)
        self.assertEqual(parsed['hidden_act'], 'gelu_pytorch_tanh')
        for change in ({'norm_epsilon':0}, {'use_bias':1}, {'hidden_act':'silu'}):
            with self.assertRaises(ValueError):
                decoder_config({**config, **change})

    def test_moe_router_and_projection_dimensions(self):
        config = self.config(model_type='olmoe',head_dim=64,num_experts=64,num_experts_per_tok=8)
        self.assertFalse(decoder_config(config)['norm_topk_prob'])
        minimal = {key:value for key,value in config.items() if key!='rms_norm_eps'}
        self.assertEqual(decoder_config(minimal)['rms_norm_eps'],1e-5)
        for change in ({'num_experts':0},{'num_experts_per_tok':65},{'norm_topk_prob':1},
                       {'clip_qkv':-1},{'head_dim':128},{'attention_bias':True}):
            with self.assertRaises(ValueError):
                decoder_config({**config, **change})

    def test_unsupported_variants_fail_before_weight_loading(self):
        for changes in ({'model_type': 'qwen3_5'}, {'model_type': 'qwen3_moe'},
                        {'rope_scaling': {'type': 'dynamic', 'factor': 2}},
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

    def test_numerical_hybrid_metadata_is_validated_before_arithmetic(self):
        for name in ('rope_theta', 'rms_norm_eps', 'partial_rotary_factor'):
            for value in (float('nan'), float('inf'), -float('inf'), True, False,
                          None, '1', [], 0, -1, 10**400):
                config = self.config()
                target = config['text_config']
                if name != 'rms_norm_eps':
                    target = target['rope_parameters']
                target[name] = value
                with self.subTest(name=name, kind=type(value).__name__, value=str(value)[:30]), \
                        self.assertRaises(ValueError):
                    decoder_config(config)
        config = self.config()
        del config['text_config']['rope_parameters']['rope_theta']
        self.assertEqual(decoder_config(config)['rope_theta'], 10000.)
        config['text_config']['rope_parameters']['partial_rotary_factor'] = 1
        config['text_config']['rms_norm_eps'] = 1
        self.assertEqual(decoder_config(config)['rotary_dim'], 256)

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


class Gpt2ConfigTests(unittest.TestCase):
    def config(self,**changes):
        return {'model_type':'gpt2','architectures':['GPT2LMHeadModel'],
                'activation_function':'gelu_new','n_embd':768,'n_head':12,
                'n_layer':12,'n_positions':1024,'vocab_size':50257,**changes}

    def test_learned_position_dimensions(self):
        c=decoder_config(self.config())
        self.assertEqual((c['head_dim'],c['intermediate_size'],c['max_position_embeddings']),(64,3072,1024))
        self.assertTrue(c['tie_word_embeddings'])
        self.assertEqual(decoder_config(self.config(n_inner=1024))['intermediate_size'],1024)

    def test_unsupported_modes_rejected(self):
        for change in ({'add_cross_attention':True},{'scale_attn_weights':False},
                       {'scale_attn_by_inverse_layer_idx':True},{'reorder_and_upcast_attn':True},
                       {'tie_word_embeddings':False},{'activation_function':'relu'},
                       {'n_inner':True},{'n_head':13},{'n_positions':0},
                       {'layer_norm_epsilon':float('nan')},{'architectures':['GPT2Model']}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                decoder_config(self.config(**change))


class YarnConfigTests(unittest.TestCase):
    def test_yarn_validation_and_attention_scaling(self):
        from dllm.rotary import attention_factor
        config = ConfigTests().config(rope_scaling={'type':'yarn', 'factor':4})
        parsed = decoder_config(config)
        self.assertGreater(attention_factor(parsed), 1)
        self.assertEqual(attention_factor({**parsed, 'rope_scaling':{**parsed['rope_scaling'], 'attention_factor':1.2}}), 1.2)
        for change in ({'factor':0}, {'beta_fast':0}, {'beta_slow':64}, {'attention_factor':float('nan')},
                       {'original_max_position_embeddings':False}, {'original_max_position_embeddings':None}, {'mscale':1}, {'truncate':1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                decoder_config({**config, 'rope_scaling':{**config['rope_scaling'], **change}})
        for change in ({'rope_theta':float('nan')}, {'rope_theta':True}, {'rope_theta':1}, {'max_position_embeddings':16, 'rope_scaling':{'type':'yarn', 'factor':2, 'original_max_position_embeddings':32}}):
            with self.assertRaises(ValueError):decoder_config({**config, **change})


class MalformedConfigTests(unittest.TestCase):
    def test_nested_containers_fail_as_configuration_errors(self):
        for value in (None, False, 1, 'layers', [], {'full_attention': 1}):
            for family in ('dense', 'hybrid-text', 'hybrid-rope', 'hybrid-layers', 'gemma'):
                if family == 'dense' and value == []: continue
                if family == 'dense': config = ConfigTests().config(layer_types=value)
                elif family == 'gemma': config = GemmaConfigTests().config(layer_types=value)
                else:
                    config = HybridConfigTests().config()
                    if family == 'hybrid-text': config['text_config'] = value
                    else: config['text_config']['rope_parameters' if family == 'hybrid-rope' else 'layer_types'] = value
                with self.subTest(family=family, kind=type(value).__name__), self.assertRaises(ValueError):
                    decoder_config(config)
        for root in (None, [], 'config', 1):
            with self.subTest(root=root), self.assertRaises(ValueError): decoder_config(root)

    def test_oversized_numeric_metadata_rejects_without_float_overflow(self):
        from tests.test_neox import config as neox
        from tests.test_bloom import config as bloom
        cases = [
            (ConfigTests().config(), ('rope_theta', 'rms_norm_eps')),
            (ConfigTests().config(model_type='starcoder2', hidden_act='gelu_pytorch_tanh'), ('norm_epsilon',)),
            (ConfigTests().config(model_type='olmoe', head_dim=64, num_experts=4, num_experts_per_tok=2), ('clip_qkv',)),
            (GemmaConfigTests().config(), ('query_pre_attn_scalar', 'rope_local_base_freq')),
            (Gpt2ConfigTests().config(), ('layer_norm_epsilon',)),
            (neox(), ('rotary_pct', 'rotary_emb_base', 'layer_norm_eps')),
            (bloom(), ('layer_norm_epsilon',)),
        ]
        for config, fields in cases:
            decoder_config(config)
            for field in fields:
                with self.subTest(family=config['model_type'], field=field), self.assertRaises(ValueError):
                    decoder_config({**config, field:10**400})
        schedules = [({'type':'linear','factor':2}, ('factor',)),
            ({'type':'llama3','factor':2,'low_freq_factor':1,'high_freq_factor':4,
              'original_max_position_embeddings':128}, ('factor','low_freq_factor','high_freq_factor')),
            ({'type':'yarn','factor':2}, ('factor','attention_factor','beta_fast','beta_slow'))]
        for config, fields in schedules:
            rotary_scaling(config)
            for field in fields:
                with self.subTest(schedule=config['type'], field=field), self.assertRaises(ValueError):
                    rotary_scaling({**config, field:10**400})
