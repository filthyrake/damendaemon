from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from dllm.deepseek_checkpoint import tensor_specs, audit_names, audit_headers, audit_snapshot


def config():
    return dict(model_type='deepseek_v4', hidden_size=32, vocab_size=16, num_hidden_layers=3,
                n_routed_experts=3, num_experts_per_tok=2, moe_intermediate_size=64, hc_mult=2,
                num_attention_heads=4, head_dim=128, q_lora_rank=32, o_groups=2, o_lora_rank=64,
                index_n_heads=2, index_head_dim=128, num_hash_layers=1, compress_ratios=[0,4,128],
                n_shared_experts=1, expert_dtype='fp4',
                quantization_config=dict(quant_method='fp8', fmt='e4m3', scale_fmt='ue8m0', weight_block_size=[128,128]))


class DeepSeekCheckpointTests(unittest.TestCase):
    def test_pinned_official_name_fingerprint_and_mtp_schedule(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/deepseek_v4_flash_contract.json').read_text())
        names = sorted(tensor_specs(fixture['config']))
        fingerprint = hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
        self.assertEqual(len(names), fixture['main_tensor_count'])
        self.assertEqual(fingerprint, fixture['main_tensor_names_sha256'])
        self.assertEqual(len(fixture['config']['compress_ratios']), 44)
        self.assertEqual(fixture['config']['num_hidden_layers'], 43)

    def test_explicit_shapes_and_per_layer_variants(self):
        specs = tensor_specs(config())
        expected = {'embed.weight':(16,32), 'hc_head_fn':(2,64),
                    'layers.0.hc_attn_fn':(8,64), 'layers.0.attn.wo_a.weight':(128,256),
                    'layers.0.attn.wo_a.scale':(1,2), 'layers.0.ffn.gate.tid2eid':(16,2),
                    'layers.1.ffn.gate.bias':(3,), 'layers.1.attn.compressor.ape':(4,256),
                    'layers.2.attn.compressor.ape':(128,128),
                    'layers.1.attn.indexer.wq_b.weight':(256,32),
                    'layers.0.ffn.experts.2.w1.weight':(64,16),
                    'layers.0.ffn.experts.2.w2.weight':(32,32),
                    'layers.0.ffn.experts.2.w2.scale':(32,2),
                    'layers.0.ffn.shared_experts.w2.weight':(32,64)}
        for name, shape in expected.items():
            self.assertEqual(specs[name].shape, shape, name)
        with_mtp = config() | {'num_nextn_predict_layers':1, 'compress_ratios':[0,4,128,0]}
        self.assertEqual(tensor_specs(with_mtp), specs)
        self.assertNotIn('layers.0.attn.compressor.ape', specs)
        self.assertNotIn('layers.2.attn.indexer.wq_b.weight', specs)
        self.assertNotIn('layers.0.ffn.gate.bias', specs)

    def test_names_missing_extras_duplicates_and_explicit_mtp_exclusion(self):
        names = list(tensor_specs(config()))
        result = audit_names(config(), names+['mtp.0.unqualified.weight'])
        self.assertEqual(result['main_tensors'], len(names))
        self.assertEqual(result['excluded_mtp'], ['mtp.0.unqualified.weight'])
        for invalid in (names[:-1], names+['unexpected.weight'], names+[names[0]], names+[None]):
            with self.assertRaises(ValueError):
                audit_names(config(), invalid)

    def test_shape_and_storage_headers_reject_wrong_packed_layout(self):
        storage = {'floating':'BF16', 'fp8':'F8_E4M3', 'fp4':'I8', 'scale':'F8_E8M0', 'integer':'I32'}
        specs = tensor_specs(config())
        headers = {name:dict(shape=list(spec.shape), dtype=storage[spec.storage]) for name, spec in specs.items()}
        self.assertEqual(audit_headers(config(), headers)['main_tensors'], len(specs))
        for name, update in (('layers.0.ffn.experts.0.w1.weight', {'shape':[64,32]}),
                             ('layers.0.ffn.experts.0.w1.scale', {'dtype':'F32'}),
                             ('layers.0.ffn.gate.tid2eid', {'dtype':'BF16'})):
            invalid = deepcopy(headers)
            invalid[name].update(update)
            with self.assertRaisesRegex(ValueError, 'shape/storage mismatch'):
                audit_headers(config(), invalid)

    def test_index_only_snapshot_does_not_require_or_open_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text(json.dumps(config()))
            names = list(tensor_specs(config()))+['mtp.0.unqualified.weight']
            index = {'weight_map':{name:'nested/weights.safetensors' for name in names}}
            (path/'model.safetensors.index.json').write_text(json.dumps(index))
            with patch('dllm.checkpoint.inspect_snapshot', side_effect=AssertionError('shard opened')):
                report = audit_snapshot(path, names_only=True)
                self.assertEqual(report['scope'], 'tensor_names_only')
                self.assertEqual(report['excluded_mtp_count'], 1)
                self.assertFalse(report['weight_values_read'])
                self.assertFalse(report['model_execution_qualified'])
                index['weight_map'][names[0]] = '../outside.safetensors'
                (path/'model.safetensors.index.json').write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    audit_snapshot(path, names_only=True)

    def test_unsupported_config_rejected_before_spec_generation(self):
        for update in ({'num_hidden_layers':True}, {'compress_ratios':[0,4]}, {'compress_ratios':[0,8,128]},
                       {'expert_dtype':'fp8'}, {'n_shared_experts':2}, {'hidden_size':33},
                       {'num_hash_layers':4}, {'n_routed_experts':1000000}, {'o_groups':3},
                       {'quantization_config':{'quant_method':'fp8'}}):
            with self.assertRaises(ValueError):
                tensor_specs(config() | update)
