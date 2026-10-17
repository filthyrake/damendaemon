from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from damendaemon.glm_flash_checkpoint import ROOT,tensor_specs,audit_names,audit_headers,audit_snapshot


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/glm53_flash_contract.json').read_text())


def config():
    c=fixture()['config'];t=c['text_config']
    t.update(hidden_size=16,vocab_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,
             q_lora_rank=8,kv_lora_rank=8,qk_nope_head_dim=4,qk_head_dim=4,v_head_dim=4,index_n_heads=2,
             index_head_dim=8,index_topk=2,index_kpool=4,intermediate_size=32,moe_intermediate_size=8,
             n_routed_experts=3,n_shared_experts=1,num_experts_per_tok=2,hc_mult=2,first_k_dense_replace=1,
             num_nextn_predict_layers=1,layer_types=['linear_attention','deepseek_sparse_attention','linear_attention'],
             mlp_layer_types=['dense','sparse','sparse'],indexer_types=['full']*3)
    t['linear_attn_config'].update(num_heads=2,head_dim=4,short_conv_kernel_size=4,kda_layers=[0,2],full_attn_layers=[1])
    exclusions=[]
    for i in (0,2):
        exclusions.extend(f'model.layers.{i}.self_attn.{n}' for n in
                          ('q_proj','k_proj','v_proj','f_a_proj','f_b_proj','g_a_proj','g_b_proj','b_proj','o_proj'))
    exclusions.extend('model.layers.1.self_attn.'+n for n in ('kv_b_proj','indexer.wq_b','indexer.wk','indexer.weights_proj'))
    c['quantization_config']['modules_to_not_convert']=exclusions
    return c


def headers(c):
    storage={'floating':'BF16','fp8':'F8_E4M3','scale':'F32'}
    return {n:dict(shape=list(s.shape),dtype=storage[s.storage]) for n,s in tensor_specs(c).items()}


class FlashCheckpointTests(unittest.TestCase):
    def test_pinned_inventory_and_distinct_attention_geometry(self):
        f=fixture();s=tensor_specs(f['config'])
        self.assertEqual(len(s),74001)
        self.assertEqual(hashlib.sha256(('\n'.join(sorted(s))+'\n').encode()).hexdigest(),f['main_tensor_names_sha256'])
        for name,shape in {'layers.0.self_attn.q_proj.weight':(8192,4096),'layers.0.hc_attn_fn':(24,16384),
                           'layers.3.self_attn.kv_b_proj.weight':(32768,512),
                           'layers.3.self_attn.kv_a_proj_with_mqa.weight':(512,4096),
                           'layers.3.self_attn.indexer.index_kpool_compress_ape':(4,128)}.items():
            self.assertEqual(s[ROOT+name].shape,shape)
        self.assertEqual((f['excluded_mtp_count'],f['excluded_vision_count']),(1760,347))

    def test_mixed_storage_and_synthetic_geometry(self):
        s=tensor_specs(config())
        for name,shape,storage in [('layers.0.self_attn.q_conv1d.weight',(8,1,4),'floating'),
                                   ('layers.0.hc_ffn_fn',(8,32),'floating'),
                                   ('layers.1.self_attn.kv_b_proj.weight',(16,8),'floating'),
                                   ('layers.1.self_attn.q_b_proj.weight',(8,8),'fp8'),
                                   ('layers.2.mlp.experts.2.down_proj.weight_scale_inv',(1,1),'scale')]:
            self.assertEqual((s[ROOT+name].shape,s[ROOT+name].storage),(shape,storage))

    def test_inventory_and_explicit_unqualified_exclusions(self):
        c=config();names=list(tensor_specs(c));mtp=ROOT+'layers.3.unqualified';vision='model.visual.unqualified'
        report=audit_names(c,names+[mtp,vision]);self.assertEqual(report['excluded_mtp'],[mtp]);self.assertEqual(report['excluded_vision'],[vision])
        for bad in (names[:-1],names+[names[0]],names+[None],names+[ROOT+'layers.4.weight'],names+['model.visualized.weight']):
            with self.assertRaises(ValueError):audit_names(c,bad)

    def test_header_storage_shape_and_integer_validation(self):
        c=config();h=headers(c);audit_headers(c,h)
        for name,change in [('layers.0.self_attn.q_proj.weight',{'dtype':'F8_E4M3'}),
                            ('layers.1.self_attn.kv_b_proj.weight',{'dtype':'F8_E4M3'}),
                            ('layers.0.mlp.up_proj.weight',{'dtype':'BF16'}),
                            ('layers.0.mlp.up_proj.weight_scale_inv',{'dtype':'F8_E8M0'}),
                            ('layers.0.hc_attn_fn',{'shape':[8,16]}),
                            ('layers.0.mlp.up_proj.weight_scale_inv',{'shape':[True,True]})]:
            bad=deepcopy(h);bad[ROOT+name].update(change)
            with self.assertRaisesRegex(ValueError,'shape/storage'):audit_headers(c,bad)

    def test_configuration_bounds_and_storage_policy(self):
        for change in ({'qk_rope_head_dim':4},{'mhc':False},{'hc_mult':17},{'hc_sinkhorn_iters':129},
                       {'num_hidden_layers':True},{'n_routed_experts':1000000},{'indexer_types':['shared']*3},
                       {'swiglu_limit':float('nan')},{'qk_head_dim':8},{'topk_group':True}):
            c=config();c['text_config'].update(change)
            with self.assertRaises(ValueError):tensor_specs(c)
        for change in ({'kda_layers':[False,2]},{'full_attn_layers':[2]},{'gate_lower_bound':float('nan')}):
            c=config();c['text_config']['linear_attn_config'].update(change)
            with self.assertRaises(ValueError):tensor_specs(c)
        for exclusions in ([],['model'],['q_proj', 'mlp']):
            c=config();c['quantization_config']['modules_to_not_convert']=exclusions
            with self.assertRaises(ValueError):tensor_specs(c)

    def test_snapshot_names_and_headers_never_read_values(self):
        from tests.test_packed_checkpoint import write_raw
        from damendaemon.safetensor_header import ITEM_BYTES
        c=config();inventory=headers(c)
        inventory[ROOT+'layers.3.unqualified']=dict(shape=[1],dtype='BF16')
        inventory['model.visual.unqualified']=dict(shape=[1],dtype='BF16')
        offset=0
        for entry in inventory.values():
            size=math.prod(entry['shape'])*ITEM_BYTES[entry['dtype']]
            entry['data_offsets']=[offset,offset+size];offset+=size
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'config.json').write_text(json.dumps(c))
            index={'weight_map':{n:'model.safetensors' for n in inventory}}
            (path/'model.safetensors.index.json').write_text(json.dumps(index))
            with patch('damendaemon.checkpoint.inspect_snapshot',side_effect=AssertionError('opened shards')):
                result=audit_snapshot(path,names_only=True)
            self.assertEqual(result['excluded_vision_count'],1);self.assertFalse(result['vision_validated'])
            write_raw(path/'model.safetensors',inventory,bytes(offset))
            with patch('damendaemon.packed_checkpoint._payload',side_effect=AssertionError('read values')):
                result=audit_snapshot(path)
            self.assertEqual(result['excluded_mtp_count'],1);self.assertFalse(result['mtp_validated'])
            self.assertFalse(result['weight_values_read']);self.assertFalse(result['model_execution_qualified'])
            index['weight_map'][next(iter(inventory))]='../outside.safetensors'
            (path/'model.safetensors.index.json').write_text(json.dumps(index))
            with self.assertRaises(ValueError):audit_snapshot(path,names_only=True)
