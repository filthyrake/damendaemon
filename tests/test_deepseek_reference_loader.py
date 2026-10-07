import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dllm.deepseek_checkpoint import tensor_specs
from dllm.deepseek_reference_loader import load_reference_model
from dllm.prepared_projection import PackedProjection
from dllm.safetensor_header import ITEM_BYTES
from tests.test_deepseek_checkpoint import config as geometry
from tests.test_packed_checkpoint import write_raw


def fixture(path):
    config=geometry()|dict(scoring_func='sqrtsoftplus',norm_topk_prob=True,topk_method='noaux_tc',hidden_act='silu',
                          tie_word_embeddings=False,sliding_window=4,hc_sinkhorn_iters=20,index_topk=2,
                          qk_rope_head_dim=64,rms_norm_eps=1e-6,hc_eps=1e-6,swiglu_limit=2.,routed_scaling_factor=1.5,
                          rope_theta=10000.,compress_rope_theta=160000.,
                          rope_scaling=dict(type='yarn',factor=2.,beta_fast=32.,beta_slow=1.,original_max_position_embeddings=16))
    rng=torch.Generator().manual_seed(173)
    headers,payload={},bytearray()
    for name,spec in tensor_specs(config).items():
        if spec.storage == 'floating':
            value=(torch.randn(spec.shape,generator=rng)*.1).to(torch.bfloat16)
            dtype='BF16'
        elif spec.storage == 'integer':
            value=torch.arange(math.prod(spec.shape)).remainder(3).reshape(spec.shape).int()
            dtype='I32'
        elif spec.storage == 'scale':
            value=torch.full(spec.shape,120,dtype=torch.uint8);dtype='F8_E8M0'
        elif spec.storage == 'fp4':
            value=torch.randint(0,256,spec.shape,generator=rng,dtype=torch.uint8);dtype='I8'
        else:
            value=torch.randint(0,119,spec.shape,generator=rng,dtype=torch.uint8);dtype='F8_E4M3'
        raw=bytes(value.contiguous().view(torch.uint8).flatten().tolist())
        start=len(payload);payload.extend(raw)
        headers[name]=dict(dtype=dtype,shape=list(spec.shape),data_offsets=[start,len(payload)])
    # Deliberately unqualified MTP payload must be inventoried but not loaded.
    headers['mtp.unqualified']=dict(dtype='U8',shape=[1],data_offsets=[len(payload),len(payload)+1]);payload.append(9)
    (path/'config.json').write_text(json.dumps(config))
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'model.safetensors' for name in headers}}))
    write_raw(path/'model.safetensors',headers,payload)
    return config,headers


class ReferenceLoaderTests(unittest.TestCase):
    def test_snapshot_builds_packed_model_and_cached_token_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers=fixture(path)
            model=load_reference_model(path)
            self.assertEqual(len(model.blocks),3)
            self.assertEqual(model.embedding.dtype,torch.float32)
            self.assertTrue(all(isinstance(block.attention_weights['wq_a.weight'],PackedProjection) for block in model.blocks))
            self.assertTrue(all(block.banks is None for block in model.blocks))
            self.assertTrue(all(block.packed_experts[0].gate.weight.dtype == torch.uint8 for block in model.blocks))
            ids=torch.tensor([[0,1,2,3,4]])
            logits,state=model.prefill(ids[:,:3],max_tokens=5)
            self.assertEqual(logits.shape,(1,16))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertGreater(float(logits.abs().max()),0)
            for position in (3,4):
                actual=model.decode(ids[:,position:position+1],state,position=position)
                expected,fresh=model.prefill(ids[:,:position+1])
                torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
                fresh.close()
            state.close()
            low=load_reference_model(path,dtype=torch.bfloat16)
            logits,state=low.prefill(ids[:,:3]);state.close()
            self.assertEqual(low.embedding.dtype,torch.bfloat16)
            self.assertEqual(low.blocks[1].attention_weights['indexer.weights_proj.weight'].dtype,torch.bfloat16)
            self.assertTrue(torch.isfinite(logits).all())

    def test_weight_and_workspace_admission_and_config_fail_before_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,_=fixture(path)
            with patch('dllm.deepseek_reference_loader._payload',side_effect=AssertionError('payload read')):
                for options in ({'max_weight_bytes':1},{'workspace_bytes':1},{'dtype':torch.int32}):
                    with self.assertRaises(ValueError):load_reference_model(path,**options)
                for change in ({'scoring_func':'sigmoid'},{'norm_topk_prob':False},{'qk_rope_head_dim':65},
                               {'tie_word_embeddings':True},{'hc_eps':float('nan')},{'index_head_dim':127}):
                    (path/'config.json').write_text(json.dumps(config|change))
                    with self.assertRaises(ValueError):load_reference_model(path)

    def test_payload_failure_closes_all_owned_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path)
            readers=[]
            def fail(reader,*args):
                readers.append(reader)
                raise OSError('injected read failure')
            with patch('dllm.deepseek_reference_loader._payload',side_effect=fail):
                with self.assertRaisesRegex(OSError,'injected'):
                    load_reference_model(path)
            self.assertTrue(readers)
            self.assertTrue(all(reader.closed for reader in readers))

    def test_exact_aggregate_parameter_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers=fixture(path)
            retained=largest=0
            for name,spec in tensor_specs(config).items():
                count=math.prod(spec.shape)
                source=count*ITEM_BYTES[headers[name]['dtype']]
                retained+=count*(4 if spec.storage in ('floating','integer') else 1)
                largest=max(largest,source)
            with patch('dllm.deepseek_reference_loader._payload',side_effect=AssertionError('payload read')):
                with self.assertRaisesRegex(ValueError,'aggregate weight budget'):
                    load_reference_model(path,max_weight_bytes=retained+largest-1)
            model=load_reference_model(path,max_weight_bytes=retained+largest)
            self.assertEqual(len(model.blocks),3)

    def test_mtp_values_skipped_and_mutated_checkpoint_rejected(self):
        from dllm import deepseek_reference_loader as loader
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);_,headers=fixture(path)
            original=loader._payload
            count=0
            def record(reader,base,entry):
                nonlocal count
                self.assertNotEqual(entry['data_offsets'],headers['mtp.unqualified']['data_offsets'])
                count+=1
                return original(reader,base,entry)
            with patch('dllm.deepseek_reference_loader._payload',side_effect=record):
                load_reference_model(path)
            self.assertEqual(count,len(headers)-1)
            def mutate(reader,base,entry):
                result=original(reader,base,entry)
                if entry['data_offsets'][0] == 0:
                    with (path/'model.safetensors').open('ab') as target:target.write(b'x')
                return result
            with patch('dllm.deepseek_reference_loader._payload',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'changed'):
                    load_reference_model(path)
