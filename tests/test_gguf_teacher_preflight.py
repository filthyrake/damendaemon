"""Teacher metadata must reject before independent HF construction/device setup."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from dllm.token_workload import token_workload_digest
from scripts import verify_gguf_reference as verifier


def fixture():
    return dict(passed=False,workload_sha256=token_workload_digest([[1]],2),cases=[
        dict(graph=graph,case=0,prompt_tokens=1,reference_tokens=[2,3]) for graph in (False,True)])


def invalid_teachers():
    values=[None,{},dict(fixture(),workload_sha256='old'),dict(fixture(),cases=[])]
    for field,value in [('graph',0),('case',True),('case',1),('prompt_tokens',True),('prompt_tokens',2),
                        ('reference_tokens',[2]),('reference_tokens',[2,True]),('reference_tokens',[2,4]),
                        ('reference_tokens',[2,-1]),('reference_tokens',[2,3.0])]:
        item=fixture();item['cases'][0][field]=value;values.append(item)
    item=fixture();item['cases'][1]=copy.deepcopy(item['cases'][0]);values.append(item)
    return values


class GgufTeacherPreflightTests(unittest.TestCase):
    def test_invalid_teacher_rejects_before_hf_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for index,teacher in enumerate(invalid_teachers()):
                with self.subTest(case=index):
                    (root/'teacher.json').write_text(json.dumps(teacher))
                    (root/'benchmark.json').write_text(json.dumps(dict(prompts={'1':[1]},output_tokens=2)))
                    args=['verify','unused.gguf','unused-config','--output',str(root/'result.json'),
                          '--teacher-trace',str(root/'teacher.json'),'--benchmark-report',str(root/'benchmark.json')]
                    source=SimpleNamespace(config={'vocab_size':4},chat_tokens=lambda *args:[1],raw_tokens=lambda *args:[1])
                    with patch('sys.argv',args),patch.object(verifier,'GgufWeights',return_value=source), \
                            patch.object(verifier.AutoConfig,'from_pretrained',side_effect=RuntimeError('HF construction boundary')) as hf:
                        with self.assertRaises(ValueError):verifier.main()
                        hf.assert_not_called()

    def test_valid_failed_diagnostic_teacher_reaches_hf_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for passed in (False,True):
                teacher=fixture();teacher['passed']=passed
                (root/'teacher.json').write_text(json.dumps(teacher))
                (root/'benchmark.json').write_text(json.dumps(dict(prompts={'1':[1]},output_tokens=2)))
                args=['verify','unused.gguf','unused-config','--output',str(root/'result.json'),
                      '--teacher-trace',str(root/'teacher.json'),'--benchmark-report',str(root/'benchmark.json')]
                source=SimpleNamespace(config={'vocab_size':4},chat_tokens=lambda *args:[1],raw_tokens=lambda *args:[1])
                with patch('sys.argv',args),patch.object(verifier,'GgufWeights',return_value=source), \
                        patch.object(verifier.AutoConfig,'from_pretrained',side_effect=RuntimeError('HF construction boundary')) as hf:
                    with self.assertRaisesRegex(RuntimeError,'HF construction boundary'):verifier.main()
                    hf.assert_called_once()
