from contextlib import redirect_stdout,redirect_stderr
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_checkpoint_pipeline import load_reference_pipeline
from dllm.glm_flash_checkpoint import tensor_specs
from dllm.glm_flash_reference_loader import load_reference_model,load_reference_stage
from dllm.safetensor_header import ITEM_BYTES
from tests.test_glm_flash_generation import text_fixture
from tests.test_glm_flash_reference_loader import fixture,floating_control
from tests.test_glm_tools import TOOLS
from dllm.glm_text import EOS
from scripts import run_glm_flash_reference as cli


class FlashCheckpointPipelineTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.text=text_fixture(self.path);self.config,self.headers,self.values=fixture(self.path,vocab_size=self.text.vocab_size)
        self.config['text_config'].update(eos_token_id=sorted(self.text.eos),pad_token_id=self.text.tokenizer.token_to_id(EOS),bos_token_id=None)
        (self.path/'config.json').write_text(json.dumps(self.config))

    def test_global_budget_before_payload_and_exact_independent_equivalence(self):
        specs=tensor_specs(self.config)
        retained=sum(math.prod(spec.shape)*(1 if spec.storage=='fp8' else 4) for spec in specs.values())
        overlap=max(math.prod(spec.shape)*ITEM_BYTES[self.headers[name]['dtype']] for name,spec in specs.items())
        with patch('dllm.glm_flash_reference_loader._payload',side_effect=AssertionError('payload')):
            for cuts,budget in (([1,2],retained+overlap-1),([3],retained+overlap),([1,1],retained+overlap),([2,1],retained+overlap),([True],retained+overlap)):
                with self.assertRaises(ValueError):load_reference_pipeline(self.path,cuts,max_weight_bytes=budget)
            with self.assertRaises(ValueError):load_reference_model(self.path,inspect_only=True)
            with self.assertRaises(ValueError):load_reference_stage(self.path,0,1,inspect_only=True)
        pipeline=load_reference_pipeline(self.path,[1,2],max_weight_bytes=retained+overlap)
        self.assertEqual(pipeline.retained_weight_bytes,retained);self.assertEqual(pipeline.weight_overlap_bytes,overlap)
        tokens=torch.tensor([[1,2,3],[3,2,1]])
        actual,state=pipeline.prefill(tokens,capacity=4,all_logits=True)
        expected,other=floating_control(self.config,self.values).prefill(tokens,capacity=4,all_logits=True)
        try:torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        finally:state.close();other.close()

    def test_receipts_reject_replacement_after_stage_or_final_construction(self):
        import dllm.glm_flash_checkpoint_pipeline as builder
        for name,target in (('config.json','load_reference_stage'),('part-0.safetensors','load_reference_stage'),
                            ('model.safetensors.index.json','load_reference_stage'),('part-1.safetensors','FlashLayerPipeline')):
            original=getattr(builder,target);calls=[]
            def mutate(*args,**kwargs):
                value=original(*args,**kwargs);calls.append(value)
                source=self.path/name;replacement=self.path/'replacement'
                replacement.write_bytes(source.read_bytes());replacement.replace(source)
                return value
            with patch.object(builder,target,mutate):
                with self.assertRaisesRegex(ValueError,'changed'):load_reference_pipeline(self.path,[1,2])
            self.assertEqual(len(calls),1)

    def test_native_command_single_ragged_and_tool_paths_match_default(self):
        rows=self.path/'rows.json';rows.write_text(json.dumps([[dict(role='user',content='Hi')],[dict(role='user',content='Longer')]]))
        tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        for mode in (['--prompt','Hi'],['--batch-messages',str(rows),'--ragged-batch'],['--prompt','Hi','--tools',str(tools)]):
            results=[]
            for extra in ([],['--layer-splits','1','2']):
                output=io.StringIO()
                with redirect_stdout(output):self.assertEqual(cli.main([str(self.path),*mode,'--context','2048','--max-new-tokens','2',*extra]),0)
                results.append([json.loads(line) for line in output.getvalue().splitlines()])
            self.assertEqual(results[0][1:],results[1][1:])
            self.assertEqual(results[1][0]['execution'],'local_cpu_pipeline')
            self.assertEqual(results[1][0]['stage_layers'],[[0,1],[1,2],[2,3]])

    def test_bad_native_request_and_cut_fail_before_payload(self):
        with patch('dllm.glm_flash_reference_loader._payload',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
            for extra in (['--layer-splits','3'],['--layer-splits','2','2'],['--layer-splits','1','--context','1']):
                with self.assertRaises(SystemExit) as error:cli.main([str(self.path),'--prompt','Hi',*extra])
                self.assertEqual(error.exception.code,2)
