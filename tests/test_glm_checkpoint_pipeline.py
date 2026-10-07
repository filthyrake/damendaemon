import io
import json
import math
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.glm_checkpoint_pipeline import load_reference_pipeline
from dllm.glm_checkpoint import tensor_specs
from dllm.glm_reference_loader import load_reference_model
from dllm.safetensor_header import ITEM_BYTES
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture
from scripts import run_glm_reference as cli


class GlmCheckpointPipelineTests(unittest.TestCase):
    def test_global_budget_before_payload_and_exact_budget_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);config,headers,_=fixture(path)
            specs=tensor_specs(config)
            retained=sum(math.prod(spec.shape)*(1 if spec.storage=='fp8' else 4) for spec in specs.values())
            overlap=max(math.prod(spec.shape)*ITEM_BYTES[headers[name]['dtype']] for name,spec in specs.items())
            with patch('dllm.glm_reference_loader._payload',side_effect=AssertionError('payload read')):
                for splits,budget in (([2],retained+overlap-1),([1],retained+overlap),([3],retained+overlap),([2,2],retained+overlap)):
                    with self.subTest(splits=splits,budget=budget),self.assertRaises(ValueError):
                        load_reference_pipeline(path,splits,max_weight_bytes=budget)
            pipeline=load_reference_pipeline(path,[2],max_weight_bytes=retained+overlap)
            self.assertEqual(pipeline.retained_weight_bytes,retained)
            self.assertEqual(pipeline.weight_overlap_bytes,overlap)
            model=load_reference_model(path)
            tokens=torch.tensor([[1,2,3],[3,2,1]])
            actual,state=pipeline.prefill(tokens,capacity=4,all_logits=True)
            expected,other=model.prefill(tokens,capacity=4,all_logits=True)
            try:torch.testing.assert_close(actual,expected)
            finally:state.close();other.close()

    def test_config_shard_or_inventory_change_during_construction_rejected(self):
        import dllm.glm_checkpoint_pipeline as builder
        for change in ('config','shard','index'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture(path)
                original=builder.load_reference_stage;calls=[]
                def mutate(*args,**kwargs):
                    stage=original(*args,**kwargs);calls.append(stage)
                    file=path/({'config':'config.json','shard':'part-1.safetensors','index':'model.safetensors.index.json'}[change])
                    # Atomic replacement changes identity even with identical bytes.
                    replacement=path/'replacement';replacement.write_bytes(file.read_bytes());replacement.replace(file)
                    return stage
                with patch.object(builder,'load_reference_stage',mutate):
                    with self.assertRaisesRegex(ValueError,'changed'):load_reference_pipeline(path,[2])
                self.assertEqual(len(calls),1)

    def test_command_single_ragged_and_tools_match_default_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);fixture(path,vocab_size=text.vocab_size)
            rows=path/'rows.json';rows.write_text(json.dumps([[dict(role='user',content='Hi')],[dict(role='user',content='Longer')]]))
            from tests.test_glm_tools import TOOLS
            tools=path/'tools.json';tools.write_text(json.dumps(TOOLS))
            for mode in (['--prompt','Hi'],['--batch-messages',str(rows),'--ragged-batch'],['--prompt','Hi','--tools',str(tools)]):
                results=[]
                for extra in ([],['--layer-splits','2']):
                    output=io.StringIO()
                    with redirect_stdout(output):cli.main([str(path),*mode,'--context','1024','--max-new-tokens','2',*extra])
                    results.append([json.loads(line) for line in output.getvalue().splitlines()])
                self.assertEqual(results[0][1:],results[1][1:])
                self.assertEqual(results[1][0]['execution'],'local_cpu_pipeline')
                self.assertEqual(results[1][0]['stage_layers'],[[0,2],[2,3]])
                self.assertFalse(results[1][0]['official_model_qualified'])

    def test_command_rejects_invalid_request_or_cut_before_loading_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);fixture(path,vocab_size=text.vocab_size)
            with patch('dllm.glm_reference_loader._payload',side_effect=AssertionError('payload read')),redirect_stderr(io.StringIO()):
                for extra in (['--layer-splits','1'],['--layer-splits','2','2'],['--layer-splits','2','--context','1']):
                    with self.subTest(extra=extra),self.assertRaises(SystemExit) as error:
                        cli.main([str(path),'--prompt','Hi',*extra])
                    self.assertEqual(error.exception.code,2)
