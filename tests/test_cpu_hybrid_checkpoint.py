"""Source checkpoint mapping against independently prepared pipeline controls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_build import build_cpu_pipeline
from dllm.quantization import quantize,quantize_q4
from tests.test_cpu_hybrid_stage import model
from tests.test_cpu_stage_plan import OPTIONS


def snapshot(path,*,dtype=torch.float32,tied=False,prefix='model.'):
    control=model();control.weights={k:v.to(dtype).float() for k,v in control.weights.items()}
    c=dict(control.config,hidden_act='silu',attn_output_gate=True,tie_word_embeddings=tied,
           rope_parameters=dict(rope_type='default',partial_rotary_factor=.5,rope_theta=10000.))
    c.pop('rotary_dim');c.pop('rope_theta');raw={}
    w=control.weights
    raw[prefix+'embed_tokens.weight']=w['embed'];raw[prefix+'norm.weight']=w['norm']-1
    if tied:w['head']=w['embed']
    else:raw['lm_head.weight']=w['head']
    for i,kind in enumerate(c['layer_types']):
        root=prefix+f'layers.{i}.'
        raw[root+'input_layernorm.weight']=w[f'{i}.norm1']-1
        raw[root+'post_attention_layernorm.weight']=w[f'{i}.norm2']-1
        gate,up=w[f'{i}.gate_up'].chunk(2)
        raw[root+'mlp.gate_proj.weight']=gate;raw[root+'mlp.up_proj.weight']=up
        raw[root+'mlp.down_proj.weight']=w[f'{i}.down']
        if kind=='full_attention':
            q,k,v=w[f'{i}.qkv'].split((64,16,16))
            for name,value in zip(('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm'),
                                  (q,k,v,w[f'{i}.o'],w[f'{i}.q_norm']-1,w[f'{i}.k_norm']-1)):
                raw[root+'self_attn.'+name+'.weight']=value
        else:
            mapping={'qkv':'in_proj_qkv.weight','z':'in_proj_z.weight','b':'in_proj_b.weight','a':'in_proj_a.weight',
                     'out':'out_proj.weight','conv':'conv1d.weight','dt':'dt_bias','A':'A_log','gated_norm':'norm.weight'}
            for short,long in mapping.items():raw[root+'linear_attn.'+long]=w[f'{i}.{short}']
    # Nonzero source norm offsets distinguish offset norms from gated norms;
    # FP16/BF16 source rounding must happen before adding the FP32 offset.
    norm_names={'norm':prefix+'norm.weight'}
    for i,kind in enumerate(c['layer_types']):
        root=prefix+f'layers.{i}.'
        norm_names.update({f'{i}.norm1':root+'input_layernorm.weight',f'{i}.norm2':root+'post_attention_layernorm.weight'})
        if kind=='full_attention':
            norm_names.update({f'{i}.{n}':root+'self_attn.'+n+'.weight' for n in ('q_norm','k_norm')})
    for short,long in norm_names.items():
        raw[long]=(torch.randn_like(raw[long])*.2).to(dtype)
        w[short]=raw[long].float()+1
    raw={k:v.to(dtype).clone().contiguous() for k,v in raw.items()}
    (path/'config.json').write_text(json.dumps(c));write_shards(path,raw)
    return control,raw


def write_shards(path,raw):
    items=list(raw.items());mapping={}
    for index in range(2):
        name=f'model-{index}.safetensors';part=dict(items[index::2]);save_file(part,str(path/name))
        mapping.update({key:name for key in part})
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))


class CpuCheckpointTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def test_sharded_source_mapping_and_actual_pipeline(self):
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            for mode in ('none','q4_0','q8_0'):
                control,_=snapshot(self.path,dtype=dtype,prefix='model.language_model.')
                control.quantization=mode
                if mode!='none':
                    for name,value in list(control.weights.items()):
                        if value.ndim==2 and name!='embed':
                            control.weights[name],control.scales[name]=(quantize_q4 if mode=='q4_0' else quantize)(value)
                with patch('dllm.device.selected_accelerator',side_effect=AssertionError('device selection')):
                    loaded=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
                    actual=build_cpu_pipeline(loaded,[2**22]*2,**OPTIONS)
                expected=build_cpu_pipeline(control,[2**22]*2,**OPTIONS)
                try:
                    self.assertEqual(set(loaded.weights),set(control.weights));self.assertEqual(set(loaded.scales),set(control.scales))
                    for name,value in control.weights.items():torch.testing.assert_close(loaded.weights[name],value,rtol=0,atol=0)
                    for name,value in control.scales.items():torch.testing.assert_close(loaded.scales[name],value,rtol=0,atol=0)
                    self.assertEqual(loaded.bytes,sum(v.numel()*v.element_size() for v in loaded.weights.values())+
                                     sum(v.numel()*v.element_size() for v in loaded.scales.values()))
                    for tokens,slot,start,final in (([1,2],0,0,False),([3,4],0,2,True),([5,6],1,0,True)):
                        a=actual.prefill_chunk(tokens,slot,start,final);b=expected.prefill_chunk(tokens,slot,start,final)
                        if final:torch.testing.assert_close(a,b,rtol=0,atol=0)
                    a,ids=actual.decode_batch([8,9],[1,0]);b,expected_ids=expected.decode_batch([8,9],[1,0])
                    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ids,expected_ids)
                finally:actual.close();expected.close()
                self.assertTrue(loaded.weights)

    def test_tied_embedding_alias_and_packed_head(self):
        for mode in ('none','q4_0','q8_0'):
            snapshot(self.path,tied=True)
            loaded=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
            if mode=='none':self.assertIs(loaded.weights['head'],loaded.weights['embed'])
            else:
                self.assertIsNot(loaded.weights['head'],loaded.weights['embed'])
                packed,scale=(quantize_q4 if mode=='q4_0' else quantize)(loaded.weights['embed'])
                torch.testing.assert_close(loaded.weights['head'],packed);torch.testing.assert_close(loaded.scales['head'],scale)
            unique={id(v):v for v in loaded.weights.values()}
            self.assertEqual(loaded.bytes,sum(v.numel()*v.element_size() for v in unique.values())+
                             sum(v.numel()*v.element_size() for v in loaded.scales.values()))

    def test_configuration_and_capacity_precede_payload(self):
        snapshot(self.path)
        for options in (dict(max_weight_bytes=1),dict(max_workspace_bytes=1),dict(max_context=65),
                        dict(max_context=True),dict(quantization='fp8')):
            with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_checkpoint(self.path,**options)
        original=json.loads((self.path/'config.json').read_text())
        for update in (dict(vision_config={}),dict(text_config={}),dict(model_type='qwen3_5'),dict(tie_word_embeddings=1),
                       dict(rms_norm_eps=float('nan')),dict(rope_parameters=None),dict(hidden_size=2**22),
                       dict(layer_types=['linear_attention']),dict(quantization_config={'quant_method':'fp8'})):
            (self.path/'config.json').write_text(json.dumps(original|update))
            with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_checkpoint(self.path,max_context=16)

    def test_inventory_shape_dtype_index_reject_before_payload(self):
        for defect in ('extra','missing','shape','dtype','index','duplicate'):
            _,raw=snapshot(self.path);name=next(iter(raw))
            if defect=='extra':raw['model.visual.extra']=torch.ones(1)
            elif defect=='missing':del raw[name]
            elif defect=='shape':raw[name]=raw[name][:1]
            elif defect=='dtype':raw[name]=raw[name].double()
            write_shards(self.path,raw)
            if defect=='index':
                p=self.path/'model.safetensors.index.json';index=json.loads(p.read_text())
                index['weight_map'][name]='model-1.safetensors';p.write_text(json.dumps(index))
            elif defect=='duplicate':save_file(raw,str(self.path/'model-1.safetensors'))
            with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_checkpoint(self.path,max_context=16)

    def test_payload_failure_nonfinite_and_mutation_close_descriptors(self):
        from dllm import cpu_hybrid_checkpoint as loader
        payload=loader._payload
        for defect in ('failure','nonfinite','mutation'):
            _,raw=snapshot(self.path)
            if defect=='nonfinite':raw[next(iter(raw))][0,0]=float('nan');write_shards(self.path,raw)
            readers=[]
            def read(reader,base,entry):
                readers.append(reader)
                if defect=='failure':raise OSError('injected payload failure')
                value=payload(reader,base,entry)
                if defect=='mutation':
                    import os
                    stamp=os.stat(reader.name);os.utime(reader.name,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000))
                return value
            with patch.object(loader,'_payload',read):
                with self.assertRaises((ValueError,OSError)):load_cpu_checkpoint(self.path,max_context=16)
            self.assertTrue(readers);self.assertTrue(all(r.closed for r in readers))

    def test_exact_admission_boundary(self):
        snapshot(self.path)
        for mode in ('none','q4_0','q8_0'):
            loaded=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
            options=dict(quantization=mode,max_context=16,max_weight_bytes=loaded.bytes,
                         max_workspace_bytes=loaded.loader_workspace_bytes)
            load_cpu_checkpoint(self.path,**options)
            for key in ('max_weight_bytes','max_workspace_bytes'):
                with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                    with self.assertRaises(ValueError):load_cpu_checkpoint(self.path,**(options|{key:options[key]-1}))
