"""Comparator gates must survive optimization; no server or model is used."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io, json, sys
from unittest.mock import patch
from scripts import compare_llama
case = sys.argv[1]
choice = {'delta': {'content': 'hello'}}
if case == 'no-output': choice['delta'] = {}
if case == 'scored': choice['logprobs'] = {'content': [{'token': 'hello'}]}
chunks = [{'choices': [choice]}]
if case.startswith('error-'):
    chunks[0]['error'] = {'empty':{}, 'false':False, 'zero':0, 'list':[], 'text':'', 'null':None}[case[6:]]
if case != 'no-usage':
    tokens = {'short': 1, 'float': 2.0, 'bool': True}.get(case, 2)
    chunks.append({'choices': [], 'usage': {'completion_tokens': tokens}})
stream = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
if case != 'no-done': stream += 'data: [DONE]\n\n'
args = ['compare', 'http://unused.invalid/v1', '--engine', 'dllm', '--max-tokens', '2']
if case in ('no-scores', 'scored'): args += ['--top-logprobs', '0']
with patch('sys.argv', args), patch.object(compare_llama, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(stream.encode())):
    compare_llama.main()
'''


class ComparatorGatesTests(unittest.TestCase):
    def run_case(self, case, optimized):
        return subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', HARNESS, case],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)

    def test_invalid_evidence_never_publishes_comparison(self):
        for optimized in (False, True):
            for case in ('no-done', 'no-output', 'no-usage', 'short', 'float', 'bool', 'no-scores',
                         'error-empty','error-false','error-zero','error-list','error-text'):
                with self.subTest(optimized=optimized, case=case):
                    result = self.run_case(case, optimized)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('ValueError', result.stderr)

    def test_complete_evidence_is_accepted_in_both_modes(self):
        for optimized in (False, True):
            for case in ('valid', 'scored', 'error-null'):
                with self.subTest(optimized=optimized, case=case):
                    result = self.run_case(case, optimized)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(len(report['trials']), 6)
                    self.assertTrue(all(row['usage']['completion_tokens'] == 2 for row in report['trials']))
