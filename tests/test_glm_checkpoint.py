from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from damendaemon.glm_checkpoint import tensor_specs,audit_names,audit_headers,audit_snapshot


def config():
    return dict(model_type='glm_moe_dsa',hidden_size=16,vocab_size=32,num_hidden_layers=3,
                num_attention_heads=2,num_key_value_heads=2,q_lora_rank=8,kv_lora_rank=8,
                qk_nope_head_dim=4,qk_rope_head_dim=4,qk_head_dim=8,v_head_dim=6,index_n_heads=2,
                index_head_dim=8,index_topk=2,intermediate_size=32,moe_intermediate_size=16,
                n_routed_experts=3,n_shared_experts=1,num_experts_per_tok=2,first_k_dense_replace=1,
                num_nextn_predict_layers=1,indexer_types=['full','shared','full'],
                mlp_layer_types=['dense','sparse','sparse'],attention_bias=False,tie_word_embeddings=False,
                hidden_act='silu',rope_interleave=True,indexer_rope_interleave=True,scoring_func='sigmoid',
                norm_topk_prob=True,topk_method='noaux_tc',n_group=1,topk_group=1,
                rope_parameters={'rope_type':'default','rope_theta':8000000},
                quantization_config={'quant_method':'fp8','fmt':'e4m3','activation_scheme':'dynamic','weight_block_size':[128,128]})


def headers(c):
    storage={'floating':'BF16','fp8':'F8_E4M3','scale':'F32'}
    return {name:dict(shape=list(spec.shape),dtype=storage[spec.storage]) for name,spec in tensor_specs(c).items()}


class GlmCheckpointTests(unittest.TestCase):
    def test_pinned_official_inventory_fingerprint(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/glm53_contract.json').read_text())
        specs=tensor_specs(fixture['config'])
        self.assertEqual(len(specs),fixture['main_tensor_count'])
        self.assertEqual(hashlib.sha256(('\n'.join(sorted(specs))+'\n').encode()).hexdigest(),fixture['main_tensor_names_sha256'])
        self.assertEqual(fixture['excluded_mtp_count'],1569)
        self.assertEqual(specs['model.layers.0.self_attn.kv_b_proj.weight'].shape,(28672,512))
        self.assertEqual(specs['model.layers.0.self_attn.q_b_proj.weight'].shape,(16384,2048))

    def test_latent_shapes_dense_sparse_and_shared_indexer_ownership(self):
        specs=tensor_specs(config())
        expected={'model.layers.0.self_attn.q_b_proj.weight':(16,8),
                  'model.layers.0.self_attn.kv_a_proj_with_mqa.weight':(12,16),
                  'model.layers.0.self_attn.kv_b_proj.weight':(20,8),
                  'model.layers.0.self_attn.o_proj.weight':(16,12),
                  'model.layers.0.self_attn.indexer.k_norm.bias':(8,),
                  'model.layers.0.self_attn.indexer.wq_b.weight':(16,8),
                  'model.layers.0.mlp.down_proj.weight':(16,32),
                  'model.layers.1.mlp.experts.2.gate_proj.weight':(16,16),
                  'model.layers.1.mlp.shared_experts.down_proj.weight_scale_inv':(1,1)}
        for name,shape in expected.items():self.assertEqual(specs[name].shape,shape,name)
        self.assertFalse(any(name.startswith('model.layers.1.self_attn.indexer.') for name in specs))
        self.assertNotIn('model.layers.0.mlp.gate.weight',specs)
        self.assertEqual(specs['model.layers.1.mlp.gate.e_score_correction_bias'].shape,(3,))

    def test_missing_extra_names_and_bounded_mtp_exclusion(self):
        c=config();names=list(tensor_specs(c))
        excluded='model.layers.3.unqualified.weight'
        self.assertEqual(audit_names(c,names+[excluded])['excluded_mtp'],[excluded])
        for invalid in (names[:-1],names+[names[0]],names+[None],names+['model.layers.4.unqualified.weight'],names+['model.layers.03.weight']):
            with self.assertRaises(ValueError):audit_names(c,invalid)

    def test_shape_and_storage_mismatches(self):
        c=config();valid=headers(c)
        self.assertEqual(audit_headers(c,valid)['main_tensors'],len(valid))
        for name,change in (('model.layers.0.self_attn.q_b_proj.weight',{'shape':[8,8]}),
                            ('model.layers.0.self_attn.q_b_proj.weight_scale_inv',{'dtype':'F8_E8M0'}),
                            ('model.layers.1.mlp.experts.0.gate_proj.weight',{'dtype':'BF16'}),
                            ('model.layers.0.input_layernorm.weight',{'shape':[True]})):
            bad=deepcopy(valid);bad[name].update(change)
            with self.assertRaisesRegex(ValueError,'shape/storage mismatch'):audit_headers(c,bad)

    def test_malformed_geometry_and_quantization_fail_before_inventory(self):
        c=config()
        for change in ({'model_type':'glm4_moe'},{'q_lora_rank':None},{'num_hidden_layers':True},
                       {'n_routed_experts':1000000},{'indexer_types':['shared']*3},
                       {'mlp_layer_types':['dense']*3},{'qk_head_dim':4},{'num_key_value_heads':1},
                       {'n_group':2},{'attention_bias':True},{'rope_interleave':False},
                       {'quantization_config':{'quant_method':'fp8'}},
                       {'quantization_config':c['quantization_config']|{'modules_to_not_convert':['model.layers.1.mlp.experts']}}):
            with self.assertRaises(ValueError):tensor_specs(c|change)

    def test_names_only_and_full_header_audits_without_tensor_reads(self):
        from tests.test_packed_checkpoint import write_raw
        from damendaemon.safetensor_header import ITEM_BYTES
        c=config();inventory=headers(c)
        offset=0
        for entry in inventory.values():
            size=math.prod(entry['shape'])*ITEM_BYTES[entry['dtype']]
            entry['data_offsets']=[offset,offset+size];offset+=size
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'config.json').write_text(json.dumps(c))
            index={'weight_map':{name:'model.safetensors' for name in inventory}}
            (path/'model.safetensors.index.json').write_text(json.dumps(index))
            with patch('damendaemon.checkpoint.inspect_snapshot',side_effect=AssertionError('shards opened')):
                self.assertEqual(audit_snapshot(path,names_only=True)['main_tensors'],len(inventory))
            write_raw(path/'model.safetensors',inventory,bytes(offset))
            with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('values read')):
                result=audit_snapshot(path)
            self.assertEqual(result['main_tensors'],len(inventory))
            self.assertFalse(result['weight_values_read']);self.assertFalse(result['model_execution_qualified'])
            index['weight_map'][next(iter(inventory))]='../outside.safetensors'
            (path/'model.safetensors.index.json').write_text(json.dumps(index))
            with self.assertRaises(ValueError):audit_snapshot(path,names_only=True)
