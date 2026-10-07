import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_reference_loader import load_reference_model
from dllm.glm_checkpoint import tensor_specs
from dllm.glm_cached_attention import CachedGlmAttention
from dllm.glm_ffn import GlmFeedForward
from dllm.glm_block import GlmBlock
from dllm.glm_model import GlmModel
from dllm.prepared_projection import PackedProjection
from dllm.safetensor_header import ITEM_BYTES
from tests.test_glm_checkpoint import config as geometry
from tests.test_packed_checkpoint import write_raw


def fixture(path):
    config=geometry()|dict(rms_norm_eps=1e-5,routed_scaling_factor=2.5)
    headers=[{},{}];payloads=[bytearray(),bytearray()];values={}
    generator=torch.Generator().manual_seed(113)
    for index,(name,spec) in enumerate(tensor_specs(config).items()):
        if spec.storage=='fp8':
            value=(torch.randn(spec.shape,generator=generator)*.5).to(torch.float8_e4m3fn).view(torch.uint8);dtype='F8_E4M3'
        elif spec.storage=='scale':value=torch.full(spec.shape,.25);dtype='F32'
        else:
            value=(torch.ones(spec.shape) if 'norm.weight' in name else torch.randn(spec.shape,generator=generator)*.2).to(torch.bfloat16);dtype='BF16'
        shard=index%2;payload=payloads[shard];start=len(payload)
        payload.extend(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
        headers[shard][name]=dict(dtype=dtype,shape=list(spec.shape),data_offsets=[start,len(payload)])
        values[name]=value
    payload=payloads[0];headers[0]['model.layers.3.unqualified']=dict(dtype='U8',shape=[1],data_offsets=[len(payload),len(payload)+1]);payload.append(9)
    (path/'config.json').write_text(json.dumps(config))
    mapping={}
    for i in range(2):
        filename=f'part-{i}.safetensors';write_raw(path/filename,headers[i],payloads[i]);mapping|={name:filename for name in headers[i]}
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    return config,headers[0]|headers[1],values


def floating_control(config,values):
    decoded={}
    for name,spec in tensor_specs(config).items():
        if spec.storage=='scale':continue
        value=values[name]
        # This fixture fits one scale block. Native torch FP8 conversion is
        # independent of the implementation's byte decoder.
        decoded[name]=(value.view(torch.float8_e4m3fn).float()*values[name[:-6]+'weight_scale_inv'][0,0]
                       if spec.storage=='fp8' else value.float())
    blocks=[]
    for layer,kind in enumerate(config['indexer_types']):
        prefix=f'model.layers.{layer}';attn=prefix+'.self_attn.';mlp=prefix+'.mlp.'
        attention=CachedGlmAttention({name[len(attn):]:value for name,value in decoded.items() if name.startswith(attn)},
                                     heads=2,nope_dim=4,rope_dim=4,value_dim=6,index_top_k=2,indexer_type=kind)
        if layer==0:weights={name[len(mlp):]:value for name,value in decoded.items() if name.startswith(mlp)}
        else:
            weights={name[len(mlp):]:value for name,value in decoded.items() if name.startswith(mlp) and not name.startswith(mlp+'experts.')}
            for name in ('gate_proj.weight','up_proj.weight','down_proj.weight'):
                weights['experts.'+name]=torch.stack([decoded[mlp+f'experts.{i}.'+name] for i in range(3)])
        ff=GlmFeedForward(weights,kind='dense' if layer==0 else 'sparse',top_k=2)
        blocks.append(GlmBlock(attention,ff,decoded[prefix+'.input_layernorm.weight'],decoded[prefix+'.post_attention_layernorm.weight']))
    return GlmModel(decoded['model.embed_tokens.weight'],blocks,decoded['model.norm.weight'],decoded['lm_head.weight'])


class GlmReferenceLoaderTests(unittest.TestCase):
    def test_indexed_snapshot_matches_independent_dense_and_cached_logits(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers,values=fixture(path)
            model=load_reference_model(path);control=floating_control(config,values)
            self.assertTrue(all(isinstance(b.attention.weights['q_a_proj.weight'],PackedProjection) for b in model.blocks))
            self.assertEqual(model.blocks[1].feed_forward.experts[0].gate.weight.dtype,torch.uint8)
            tokens=torch.tensor([[1,4,3,2],[2,3,1,5]])
            expected,other=control.prefill(tokens,capacity=4,all_logits=True);self.addCleanup(other.close)
            first,state=model.prefill(tokens[:,:2],capacity=4,all_logits=True);self.addCleanup(state.close)
            torch.testing.assert_close(first,expected[:,:2],rtol=3e-5,atol=3e-6)
            for position in (2,3):
                actual=model.forward(tokens[:,position:position+1],state,position=position)
                torch.testing.assert_close(actual,expected[:,position],rtol=3e-5,atol=3e-6)

    def test_exact_weight_budget_and_workspace_config_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers,_=fixture(path)
            retained=largest=0
            for name,spec in tensor_specs(config).items():
                count=math.prod(spec.shape);retained+=count*(1 if spec.storage=='fp8' else 4)
                largest=max(largest,count*ITEM_BYTES[headers[name]['dtype']])
            with patch('dllm.glm_reference_loader._payload',side_effect=AssertionError('values read')):
                for options in ({'max_weight_bytes':retained+largest-1},{'projection_workspace_bytes':1},{'max_workspace_bytes':1}):
                    with self.assertRaises(ValueError):load_reference_model(path,**options)
                for change in ({'rms_norm_eps':float('nan')},{'routed_scaling_factor':0},{'index_topk':4097}):
                    (path/'config.json').write_text(json.dumps(config|change))
                    with self.assertRaises(ValueError):load_reference_model(path)
            (path/'config.json').write_text(json.dumps(config))
            self.assertEqual(len(load_reference_model(path,max_weight_bytes=retained+largest).blocks),3)
            with patch('dllm.glm_reference_loader.sys.byteorder','big'):
                with self.assertRaisesRegex(ValueError,'endian'):load_reference_model(path)

    def test_mtp_omitted_payload_failure_closes_descriptors_and_mutation_rejected(self):
        import dllm.glm_reference_loader as loader
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,_,_=fixture(path);opened=[];read=[]
            original_header=loader.read_header;original_payload=loader._payload
            def header(reader,**options):opened.append(reader);return original_header(reader,**options)
            def payload(reader,base,entry):read.append(entry);return original_payload(reader,base,entry)
            with patch.object(loader,'read_header',side_effect=header),patch.object(loader,'_payload',side_effect=payload):
                load_reference_model(path)
            self.assertEqual(len(read),len(tensor_specs(config)));self.assertTrue(all(r.closed for r in opened))
            opened.clear()
            with patch.object(loader,'read_header',side_effect=header),patch.object(loader,'_payload',side_effect=OSError('injected')):
                with self.assertRaisesRegex(OSError,'injected'):load_reference_model(path)
            self.assertEqual(len(opened),2);self.assertTrue(all(r.closed for r in opened))
            with patch.object(loader,'file_stamp',return_value=('changed',)):
                with self.assertRaisesRegex(ValueError,'changed'):load_reference_model(path)

    def test_header_storage_index_and_duplicate_config_rejected_before_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,_,_=fixture(path)
            with patch('dllm.glm_reference_loader._payload',side_effect=AssertionError('values read')):
                index_path=path/'model.safetensors.index.json';index=json.loads(index_path.read_text())
                index['weight_map']['model.embed_tokens.weight']='part-1.safetensors';index_path.write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError,'index'):load_reference_model(path)
                fixture(path)
                raw=(path/'part-0.safetensors').read_bytes().replace(b'F8_E4M3',b'F8_E5M2')
                (path/'part-0.safetensors').write_bytes(raw)
                with self.assertRaisesRegex(ValueError,'storage'):load_reference_model(path)
                (path/'config.json').write_text('{"model_type":"x","model_type":"y"}')
                with self.assertRaises(ValueError):load_reference_model(path)

    def test_index_read_is_bounded_even_if_file_grows_after_stat(self):
        import io
        from dllm.checkpoint import snapshot_shards
        class GrowingIndex(io.BytesIO):
            def read(self,size=-1):
                if size!=16*1024**2+1:raise AssertionError('unbounded metadata read')
                return b' ' * size
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path);original=Path.open
            def opened(file,*args,**kwargs):
                if file.name=='model.safetensors.index.json':return GrowingIndex()
                return original(file,*args,**kwargs)
            with patch.object(Path,'open',new=opened):
                with self.assertRaisesRegex(ValueError,'metadata limit'):snapshot_shards(path)
