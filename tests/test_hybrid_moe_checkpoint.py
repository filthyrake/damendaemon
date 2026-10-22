import hashlib
import json
from pathlib import Path
import unittest

from damendaemon.hybrid_moe_checkpoint import DTYPES, audit_headers, audit_names, tensor_specs


def config():
    text = dict(model_type='qwen3_5_moe_text', hidden_size=32, vocab_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4, full_attention_interval=2,
                num_experts=3, num_experts_per_tok=2, moe_intermediate_size=8,
                shared_expert_intermediate_size=8, mtp_num_hidden_layers=1,
                attention_bias=False, attn_output_gate=True, hidden_act='silu',
                mamba_ssm_dtype='float32', rms_norm_eps=1e-6, dtype='bfloat16',
                partial_rotary_factor=0.5,
                layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
                rope_parameters=dict(mrope_interleaved=True, mrope_section=[2, 1, 1]))
    return dict(model_type='qwen3_5_moe', architectures=['Qwen3_5MoeForConditionalGeneration'],
               tie_word_embeddings=False, vision_config=dict(depth=2), text_config=text)


class HybridMoeCheckpointTests(unittest.TestCase):
    def test_synthetic_inventory_counts_and_shapes(self):
        specs = tensor_specs(config())
        # 4 layers x (2 norms + 7 MoE FFN) + 2 linear x 9 + 2 full x 6 + root 3
        self.assertEqual(len(specs), 4 * 9 + 2 * 9 + 2 * 6 + 3)
        expected = {
            'model.language_model.embed_tokens.weight': (64, 32),
            'model.language_model.norm.weight': (32,),
            'lm_head.weight': (64, 32),
            'model.language_model.layers.0.linear_attn.in_proj_qkv.weight': (64, 32),
            'model.language_model.layers.0.linear_attn.in_proj_z.weight': (32, 32),
            'model.language_model.layers.0.linear_attn.in_proj_a.weight': (4, 32),
            'model.language_model.layers.0.linear_attn.out_proj.weight': (32, 32),
            'model.language_model.layers.0.linear_attn.conv1d.weight': (64, 1, 4),
            'model.language_model.layers.0.linear_attn.dt_bias': (4,),
            'model.language_model.layers.0.linear_attn.A_log': (4,),
            'model.language_model.layers.0.linear_attn.norm.weight': (8,),
            'model.language_model.layers.1.self_attn.q_proj.weight': (64, 32),
            'model.language_model.layers.1.self_attn.k_proj.weight': (16, 32),
            'model.language_model.layers.1.self_attn.o_proj.weight': (32, 32),
            'model.language_model.layers.1.self_attn.q_norm.weight': (16,),
            'model.language_model.layers.2.mlp.gate.weight': (3, 32),
            'model.language_model.layers.2.mlp.experts.gate_up_proj': (3, 16, 32),
            'model.language_model.layers.2.mlp.experts.down_proj': (3, 32, 8),
            'model.language_model.layers.2.mlp.shared_expert.gate_proj.weight': (8, 32),
            'model.language_model.layers.2.mlp.shared_expert.down_proj.weight': (32, 8),
            'model.language_model.layers.2.mlp.shared_expert_gate.weight': (1, 32),
        }
        for name, shape in expected.items():
            self.assertEqual(specs[name].shape, shape, name)
            self.assertEqual(specs[name].storage, 'floating')
        for name in specs:
            self.assertTrue(name.startswith(('model.language_model.', 'lm_head.')), name)

    def test_mtp_setting_does_not_change_inventory(self):
        base = tensor_specs(config())
        changed = config()
        changed['text_config']['mtp_num_hidden_layers'] = 0
        self.assertEqual(tensor_specs(changed), base)

    def test_invalid_configurations_rejected(self):
        def reject(mutate, message):
            broken = config()
            mutate(broken)
            with self.assertRaises(ValueError, msg=message):
                tensor_specs(broken)
        reject(lambda c: c.update(model_type='qwen3_5'), 'root model type')
        reject(lambda c: c['text_config'].update(model_type='qwen3_5_text'), 'text model type')
        reject(lambda c: c.pop('vision_config'), 'missing vision container')
        reject(lambda c: c.update(tie_word_embeddings=True), 'tied head')
        reject(lambda c: c['text_config'].pop('layer_types'), 'missing schedule')
        reject(lambda c: c['text_config'].update(layer_types=['full_attention'] * 4), 'schedule mismatch')
        reject(lambda c: c['text_config'].update(num_experts_per_tok=4), 'top exceeds experts')
        reject(lambda c: c['text_config'].update(num_key_value_heads=4), 'kv exceeds heads')
        reject(lambda c: c['text_config'].update(attn_output_gate=False), 'output gate off')
        reject(lambda c: c['text_config'].update(dtype='float16'), 'dtype')
        reject(lambda c: c['text_config'].update(mamba_ssm_dtype='float16'), 'state dtype')
        reject(lambda c: c['text_config'].update(partial_rotary_factor=0.75), 'mrope coverage')
        reject(lambda c: c['text_config']['rope_parameters'].update(mrope_interleaved=False), 'mrope flag')
        reject(lambda c: c['text_config'].update(mtp_num_hidden_layers=17), 'mtp bound')

    def test_audit_names_exclusions_and_rejections(self):
        specs = tensor_specs(config())
        names = list(specs) + ['mtp.fc.weight', 'model.visual.patch.weight']
        report = audit_names(config(), names)
        self.assertEqual(report['main_tensors'], len(specs))
        self.assertEqual(report['excluded_mtp'], ['mtp.fc.weight'])
        self.assertEqual(report['excluded_vision'], ['model.visual.patch.weight'])
        with self.assertRaises(ValueError):
            audit_names(config, names + ['mtp.fc.weight'])
        with self.assertRaises(ValueError):
            audit_names(config, names + ['model.language_model.unexpected.weight'])
        with self.assertRaises(ValueError):
            audit_names(config, list(specs)[:-1] + ['mtp.fc.weight', 'model.visual.patch.weight'])

    def test_audit_headers_shape_and_dtype(self):
        specs = tensor_specs(config())
        headers = {name: dict(shape=list(spec.shape), dtype='BF16') for name, spec in specs.items()}
        headers['mtp.fc.weight'] = dict(shape=[32, 32], dtype='BF16')
        headers['model.visual.patch.weight'] = dict(shape=[8, 8], dtype='BF16')
        report = audit_headers(config(), headers)
        self.assertEqual(report['main_tensors'], len(specs))
        headers['lm_head.weight'] = dict(shape=[2, 2], dtype='BF16')
        with self.assertRaises(ValueError):
            audit_headers(config(), headers)
        headers['lm_head.weight'] = dict(shape=[64, 32], dtype='INT8')
        with self.assertRaises(ValueError):
            audit_headers(config(), headers)
        self.assertEqual(DTYPES['floating'], frozenset(('BF16', 'F32')))

    def test_pinned_real_checkpoint_fingerprint(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/qwen3_5_moe_contract.json').read_text())
        specs = tensor_specs(fixture['config'])
        names = sorted(specs)
        fingerprint = hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest()
        self.assertEqual(len(names), fixture['main_tensor_count'])
        self.assertEqual(fingerprint, fixture['main_tensor_names_sha256'])
        text = fixture['config']['text_config']
        self.assertEqual(text['num_hidden_layers'], 40)
        self.assertEqual(text['num_experts'], 256)
        self.assertEqual(text['num_experts_per_tok'], 8)
        self.assertEqual(text['full_attention_interval'], 4)
        self.assertEqual(text['layer_types'].count('full_attention'), 10)
        self.assertEqual(fixture['shard_count'], 26)
        self.assertEqual(fixture['total_index_tensors'], 1045)
        self.assertEqual(fixture['excluded_mtp_count'], 19)
        self.assertEqual(fixture['excluded_vision_count'], 333)
        # Prefix independently measured from the real checkpoint's config.json.
        self.assertTrue(fixture['config_sha256'].startswith('93a4693fa9d8392f'))


if __name__ == '__main__':
    unittest.main()
