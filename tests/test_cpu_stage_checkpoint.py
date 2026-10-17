from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.cpu_hybrid_checkpoint import load_cpu_checkpoint,load_cpu_stage_checkpoint
from damendaemon.cpu_hybrid_stage import CpuHybridStage
from damendaemon.cpu_pipeline_build import build_cpu_pipeline
from damendaemon.safetensor_header import read_header
from tests.test_cpu_hybrid_checkpoint import snapshot,write_shards
from tests.test_cpu_stage_plan import OPTIONS


class CpuStageCheckpointTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def test_selected_payload_names_storage_and_split_execution(self):
        from damendaemon import cpu_hybrid_checkpoint as loader
        payload=loader._payload
        for mode in ('none','q4_0','q8_0'):
            for tied in (False,True):
                snapshot(self.path,tied=tied,dtype=torch.bfloat16)
                full=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
                offsets={}
                for file in self.path.glob('*.safetensors'):
                    with file.open('rb') as reader:
                        entries,_,_=read_header(reader)
                        offsets.update({(str(file),tuple(e['data_offsets'])):name for name,e in entries.items()})
                selected=[];stages=[];reads=[]
                def read(reader,base,entry):
                    reads.append(offsets[(reader.name,tuple(entry['data_offsets']))]);return payload(reader,base,entry)
                for start,end in ((0,1),(1,3),(3,4)):
                    reads.clear()
                    with patch.object(loader,'_payload',read):
                        part=load_cpu_stage_checkpoint(self.path,start,end,quantization=mode,max_context=16)
                    self.assertEqual(part.layer_range,(start,end))
                    expected_names={n for n in full.weights if ('.' in n and start<=int(n.split('.')[0])<end)
                                    or (n=='embed' and start==0) or (n in ('head','norm') and end==4)}
                    self.assertEqual(set(part.weights),expected_names)
                    for name in expected_names:torch.testing.assert_close(part.weights[name],full.weights[name],rtol=0,atol=0)
                    for name in part.scales:torch.testing.assert_close(part.scales[name],full.scales[name],rtol=0,atol=0)
                    expected_reads={name for name in offsets.values() if (name.startswith('model.layers.') and start<=int(name.split('.')[2])<end)
                                    or (name=='model.embed_tokens.weight' and (start==0 or (end==4 and tied)))
                                    or (name in ('lm_head.weight','model.norm.weight') and end==4)}
                    self.assertEqual(set(reads),expected_reads);self.assertEqual(len(reads),len(set(reads)))
                    actual_bytes=sum(v.numel()*v.element_size() for v in part.weights.values())+sum(v.numel()*v.element_size() for v in part.scales.values())
                    self.assertEqual(part.bytes,actual_bytes);self.assertLess(part.bytes,full.bytes)
                    selected.append(part);stages.append(CpuHybridStage(part,start,end,**OPTIONS))
                control=CpuHybridStage(full,0,4,**OPTIONS)
                try:
                    for ids,pos,slots,prefill,prefix in (([[1,2],[3,4]],[[0,1],[0,1]],[0,1],True,0),
                            ([[5],[6]],[[2],[2]],[0,1],True,2),([[7],[8]],[[3],[3]],[1,0],False,0)):
                        ids=torch.tensor(ids);pos=torch.tensor(pos);slots=torch.tensor(slots)
                        actual=stages[0].embedding(ids)
                        for stage in stages:actual=stage.forward_layers(actual,pos,slots,prefill,prefix)
                        expected=control.forward_layers(control.embedding(ids),pos,slots,prefill,prefix)
                        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
                        torch.testing.assert_close(stages[-1].output_logits(actual[:,-1]),control.output_logits(expected[:,-1]),rtol=2e-4,atol=2e-6)
                finally:
                    control.close()
                    for stage in stages:stage.close()
                self.assertTrue(all(part.weights for part in selected))

    def test_selected_budget_can_fit_when_whole_model_cannot(self):
        snapshot(self.path,tied=True)
        for mode in ('none','q4_0','q8_0'):
            part=load_cpu_stage_checkpoint(self.path,1,2,quantization=mode,max_context=16)
            options=dict(quantization=mode,max_context=16,max_weight_bytes=part.bytes,max_workspace_bytes=part.loader_workspace_bytes)
            load_cpu_stage_checkpoint(self.path,1,2,**options)
            with patch('damendaemon.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_checkpoint(self.path,**options)
                for key in ('max_weight_bytes','max_workspace_bytes'):
                    with self.assertRaises(ValueError):load_cpu_stage_checkpoint(self.path,1,2,**(options|{key:options[key]-1}))
            with self.assertRaises(ValueError):build_cpu_pipeline(part,[2**22]*2,**OPTIONS)

    def test_excluded_headers_still_validated_but_values_are_not_read(self):
        _,raw=snapshot(self.path);name='model.layers.3.mlp.down_proj.weight'
        for defect in ('missing','shape','dtype'):
            changed=dict(raw)
            if defect=='missing':del changed[name]
            elif defect=='shape':changed[name]=raw[name][:1]
            else:changed[name]=raw[name].double()
            write_shards(self.path,changed)
            with patch('damendaemon.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_stage_checkpoint(self.path,0,1,max_context=16)
        raw[name][0,0]=float('nan');write_shards(self.path,raw)
        load_cpu_stage_checkpoint(self.path,0,1,max_context=16)
        with self.assertRaisesRegex(ValueError,'nonfinite'):load_cpu_stage_checkpoint(self.path,3,4,max_context=16)
        with self.assertRaisesRegex(ValueError,'nonfinite'):load_cpu_checkpoint(self.path,max_context=16)

    def test_invalid_ranges_and_whole_range_tied_alias(self):
        snapshot(self.path,tied=True)
        for start,end in ((True,2),(-1,2),(2,2),(2,1),(0,5)):
            with patch('damendaemon.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')):
                with self.assertRaises(ValueError):load_cpu_stage_checkpoint(self.path,start,end,max_context=16)
        part=load_cpu_stage_checkpoint(self.path,0,4,max_context=16)
        self.assertIs(part.weights['head'],part.weights['embed'])
