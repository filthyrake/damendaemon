"""Explicit benchmark overrides must never silently select default workloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import verify_gguf_reference as verifier


class GgufBenchmarkPreflightTests(unittest.TestCase):
    def test_malformed_override_rejects_before_source_loading(self):
        valid={'prompts':{'1':[1]},'output_tokens':2}
        invalid=[None,{},[],False,'benchmark',dict(valid,prompts={}),dict(valid,prompts=[]),
                 dict(valid,output_tokens=True),dict(valid,output_tokens=0),dict(valid,output_tokens=257)]
        for prompts in ({'bad':[1]},{'1':[]},{'1':[True]},{'1':[-1]},{'1':'bad'},
                        {'1':[1]*512},{str(i):[1] for i in range(65)}):
            invalid.append(dict(valid,prompts=prompts))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'benchmark.json'
            for value in invalid:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    args=['verify','unused.gguf','unused-config','--output',str(Path(tmp)/'report.json'),
                          '--benchmark-report',str(path)]
                    with patch('sys.argv',args),patch.object(verifier,'GgufWeights',side_effect=RuntimeError('source load boundary')) as source:
                        with self.assertRaises(ValueError):verifier.main()
                        source.assert_not_called()

    def test_valid_override_reaches_source_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'benchmark.json';path.write_text(json.dumps({'prompts':{'1':[1]},'output_tokens':2}))
            args=['verify','unused.gguf','unused-config','--output',str(Path(tmp)/'report.json'),'--benchmark-report',str(path)]
            with patch('sys.argv',args),patch.object(verifier,'GgufWeights',side_effect=RuntimeError('source load boundary')) as source:
                with self.assertRaisesRegex(RuntimeError,'source load boundary'):verifier.main()
                source.assert_called_once()
