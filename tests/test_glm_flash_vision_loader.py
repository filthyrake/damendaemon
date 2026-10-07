import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from dllm.glm_flash_vision_loader import load_reference_vision
from dllm.glm_flash_vision import FlashVision,ROOT
from tests.test_glm_flash_vision import fixture


def snapshot(path,split=True):
    c,w=fixture();config=dict(model_type='glm5_next',vision_config=c,text_config=dict(hidden_size=c['out_hidden_size']),
                            quantization_config=dict(modules_to_not_convert=['model.visual']))
    parts=[{},{}] if split else [{}];values={}
    for index,(name,value) in enumerate(w.items()):
        value=value.to((torch.float16,torch.bfloat16,torch.float32)[index%3]);parts[index%len(parts)][ROOT+name]=value;values[name]=value.float()
    parts[0]['model.language_model.unqualified']=torch.tensor([11],dtype=torch.uint8)
    (path/'config.json').write_text(json.dumps(config));mapping={}
    for index,part in enumerate(parts):
        name=f'part-{index}.safetensors';save_file(part,path/name);mapping.update({key:name for key in part})
    (path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
    return c,values,config


class FlashVisionLoaderTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def test_selected_values_only_and_owned_mixed_float_buffers(self):
        from dllm import glm_flash_vision_loader as loader
        for split in (False,True,None):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp);c,values,_=snapshot(path,split);readers=[];payload=loader._payload;header=loader.read_header
                if split is None:
                    (path/'model.safetensors.index.json').unlink();(path/'part-0.safetensors').rename(path/'model.safetensors')
                def record(reader,**options):readers.append(reader);return header(reader,**options)
                def selected(reader,base,entry):
                    self.assertNotEqual(entry['dtype'],'U8');return payload(reader,base,entry)
                with patch.object(loader,'read_header',side_effect=record),patch.object(loader,'_payload',side_effect=selected):
                    model=load_reference_vision(path)
                self.assertTrue(all(reader.closed for reader in readers))
                for name,value in values.items():torch.testing.assert_close(model.w[name],value,rtol=0,atol=0)
                patches=torch.randn(8,24)
                torch.testing.assert_close(model.forward(patches,[(1,2,4)]),FlashVision(c,values).forward(patches,[(1,2,4)]),rtol=0,atol=0)

    def test_bad_metadata_and_weight_budget_precede_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);_,_,config=snapshot(path)
            with patch('dllm.glm_flash_vision_loader._payload',side_effect=AssertionError('read payload')):
                with self.assertRaisesRegex(ValueError,'budget'):load_reference_vision(path,max_weight_bytes=1)
                with self.assertRaisesRegex(ValueError,'workspace'):load_reference_vision(path,max_workspace_bytes=1)
                config['quantization_config']['modules_to_not_convert']=[];(path/'config.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):load_reference_vision(path)
                snapshot(path);index=json.loads((path/'model.safetensors.index.json').read_text());index['weight_map'].pop(next(iter(index['weight_map'])))
                (path/'model.safetensors.index.json').write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError,'index'):load_reference_vision(path)
                snapshot(path);save_file({ROOT+'invalid':torch.zeros(1)},path/'part-1.safetensors')
                with self.assertRaises(ValueError):load_reference_vision(path)

    def test_payload_failure_and_mutation_close_descriptors(self):
        from dllm import glm_flash_vision_loader as loader
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);readers=[];header=loader.read_header;payload=loader._payload
            def record(reader,**options):readers.append(reader);return header(reader,**options)
            def mutate(reader,base,entry):
                value=payload(reader,base,entry)
                with (path/'part-0.safetensors').open('ab') as writer:writer.write(b'x')
                return value
            def nonfinite(reader,base,entry):return payload(reader,base,entry).fill_(255)
            for error in (OSError('injected read failure'),mutate,nonfinite):
                snapshot(path)
                with patch.object(loader,'read_header',side_effect=record),patch.object(loader,'_payload',side_effect=error):
                    with self.assertRaises((OSError,ValueError)):load_reference_vision(path)
                self.assertTrue(all(reader.closed for reader in readers))
