import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.deepseek_layer_stage import DeepSeekLayerStage
from damendaemon.deepseek_reference_loader import load_reference_model,load_reference_stage
from damendaemon.deepseek_checkpoint import tensor_specs
from damendaemon.safetensor_header import ITEM_BYTES
from tests.test_deepseek_reference_loader import fixture
from tests.test_deepseek_checkpoint import config as geometry


class DeepSeekLayerStageTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.config,self.headers=fixture(self.path)

    def stages(self,ranges):
        stages=[load_reference_stage(self.path,a,b) for a,b in ranges]
        for stage in stages:self.addCleanup(stage.cleanup)
        return stages

    def prefill(self,stages,tokens,capacity=140):
        hidden=None;states=[]
        for stage in stages:
            hidden,state=stage.prefill(tokens,hidden=hidden,max_tokens=capacity);states.append(state);self.addCleanup(state.close)
            if not stage.last:self.assertEqual(tuple(hidden.shape),(*tokens.shape,stage.streams,stage.dim))
        return hidden,states

    def test_two_three_stage_prefill_decode_reorder_cross_window_and_compression_boundaries(self):
        control=load_reference_model(self.path)
        for ranges,length in ((((0,1),(1,3)),3),(((0,2),(2,3)),127),(((0,1),(1,2),(2,3)),7)):
            stages=self.stages(ranges);tokens=torch.arange(2*(length+2)).reshape(2,length+2)%16
            actual,states=self.prefill(stages,tokens[:,:length])
            expected,state=control.prefill(tokens[:,:length],max_tokens=140);state.close()
            torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
            order=torch.tensor([1,0,1]);tokens=tokens[order]
            for state in states:state.reorder(order)
            for position in (length,length+1):
                hidden=None
                for stage,state in zip(stages,states):hidden=stage.decode(tokens[:,position:position+1],state,position=position,hidden=hidden)
                expected,state=control.prefill(tokens[:,:position+1],max_tokens=140);state.close()
                torch.testing.assert_close(hidden,expected,rtol=4e-5,atol=4e-6)

    def test_selected_payloads_exact_weight_budget_and_unowned_header_audit(self):
        import damendaemon.deepseek_reference_loader as loader
        specs=tensor_specs(self.config);original=loader._payload
        for start,end in ((0,1),(1,2),(2,3)):
            names={n for n in specs if n.startswith(tuple(f'layers.{i}.' for i in range(start,end)))}
            if start==0:names.add('embed.weight')
            if end==3:names|={'head.weight','norm.weight','hc_head_fn','hc_head_scale','hc_head_base'}
            budget=sum(math.prod(specs[n].shape)*(4 if specs[n].storage in ('floating','integer') else 1) for n in names)
            budget+=max(math.prod(specs[n].shape)*ITEM_BYTES[self.headers[n]['dtype']] for n in names)
            with patch.object(loader,'_payload',side_effect=AssertionError('payload')):
                with self.assertRaisesRegex(ValueError,'weight budget'):load_reference_stage(self.path,start,end,max_weight_bytes=budget-1)
            offsets={tuple(self.headers[n]['data_offsets']) for n in names};seen=[]
            def read(reader,base,entry):
                self.assertIn(tuple(entry['data_offsets']),offsets);seen.append(entry);return original(reader,base,entry)
            with patch.object(loader,'_payload',read):stage=load_reference_stage(self.path,start,end,max_weight_bytes=budget)
            self.assertEqual(len(seen),len(names));self.assertEqual(stage.embedding is not None,start==0);self.assertEqual(stage.head is not None,end==3)
            stage.cleanup()
        audit=loader.audit_headers
        def corrupt(config,headers):
            headers=dict(headers);headers.pop('head.weight');return audit(config,headers)
        with patch.object(loader,'audit_headers',corrupt),patch.object(loader,'_payload',side_effect=AssertionError('payload')):
            with self.assertRaises(ValueError):load_reference_stage(self.path,0,1)

    def test_token_geometry_cache_and_output_preflight_preserve_state(self):
        stage=self.stages(((0,2),))[0];tokens=torch.tensor([[1,2,3]])
        reserved=stage.cache_bytes_per_slot(8)
        with patch.object(stage.blocks[0],'prefill',side_effect=AssertionError('allocate')):
            with self.assertRaisesRegex(ValueError,'cache budget'):stage.prefill(tokens,max_tokens=8,max_cache_bytes=reserved-1)
            for ids in (tokens.double(),tokens[:,:0],torch.tensor([[-1]]),torch.tensor([[16]])):
                with self.assertRaises(ValueError):stage.prefill(ids,max_tokens=8)
            limit=stage.max_output_bytes;stage.max_output_bytes=4*tokens.numel()*stage.streams*stage.dim-1
            with self.assertRaisesRegex(ValueError,'output budget'):stage.prefill(tokens,max_tokens=8)
            stage.max_output_bytes=limit
        hidden,state=stage.prefill(tokens,max_tokens=8,max_cache_bytes=reserved);self.addCleanup(state.close)
        self.assertEqual(sum(layer.attention.bytes_per_slot for layer in state.layers),reserved)
        with self.assertRaises(ValueError):state.reorder(torch.tensor([0,0]))
        with self.assertRaises(ValueError):state.reorder(torch.zeros(129,dtype=torch.int64))
        for ids,position in ((torch.tensor([[16]]),3),(torch.tensor([[1]]),2)):
            with self.assertRaises(ValueError):stage.decode(ids,state,position=position)
        self.assertEqual(state.position,3);self.assertFalse(state.poisoned)
        with self.assertRaises(ValueError):load_reference_stage(self.path,0,True)
        from damendaemon.deepseek_generation import ReferenceGenerator
        with self.assertRaises(ValueError):ReferenceGenerator(stage,None)

    def test_hash_routing_on_later_stage_requires_original_tokens_and_preserves_full_streams(self):
        with patch('tests.test_deepseek_reference_loader.geometry',return_value=geometry()|dict(num_hash_layers=2)):
            fixture(self.path)
        stages=self.stages(((0,1),(1,2),(2,3)));tokens=torch.tensor([[1,2,3,4]])
        actual,_=self.prefill(stages,tokens,8);control=load_reference_model(self.path)
        expected,state=control.prefill(tokens,max_tokens=8);state.close();torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        hidden,first=stages[0].prefill(tokens,max_tokens=8);first.close()
        with patch.object(stages[1].blocks[0],'prefill',side_effect=AssertionError('allocate')):
            with self.assertRaises(ValueError):stages[1].prefill(torch.full_like(tokens,16),hidden=hidden,max_tokens=8)
            with self.assertRaises(ValueError):stages[1].prefill(tokens,hidden=hidden.mean(2),max_tokens=8)
        last=stages[-1];values=torch.zeros(1,2,2,32)
        result,state=last.prefill_activations(values,tokens[:,:2],max_tokens=8);self.addCleanup(state.close)
        self.assertEqual(tuple(result.shape),(1,2,2,32))
        result=last.decode_activations(values[:,:1],tokens[:,2:3],state,position=2)
        self.assertEqual(tuple(result.shape),(1,1,2,32))

    def test_partial_decode_and_reorder_failures_poison_owner(self):
        stage=self.stages(((0,2),))[0];_,state=stage.prefill(torch.tensor([[1,2]]),max_tokens=8);self.addCleanup(state.close)
        with patch.object(stage.blocks[-1],'decode',side_effect=RuntimeError('late')):
            with self.assertRaisesRegex(RuntimeError,'late'):stage.decode(torch.tensor([[3]]),state,position=2)
        self.assertTrue(state.poisoned);self.assertTrue(all(layer.attention.poisoned for layer in state.layers))
        _,other=stage.prefill(torch.tensor([[1,2]]),max_tokens=8);self.addCleanup(other.close)
        with patch.object(other.layers[-1],'reorder',side_effect=RuntimeError('reorder')):
            with self.assertRaisesRegex(RuntimeError,'reorder'):other.reorder(torch.tensor([0,0]))
        self.assertTrue(other.poisoned)
