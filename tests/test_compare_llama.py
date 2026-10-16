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
if case == 'scored' or case.startswith('score-'):
    entry = {'token':'hello','logprob':-.5,'bytes':[104,101,108,108,111],'top_logprobs':[]}
    content = [entry]
    if case == 'score-missing': entry.pop('logprob')
    if case == 'score-string': content = 'not score records'
    if case == 'score-entry': content = ['token']
    if case == 'score-token': entry['token'] = 1
    if case == 'score-bool': entry['logprob'] = True
    if case == 'score-positive': entry['logprob'] = .1
    if case == 'score-nan': entry['logprob'] = float('nan')
    if case == 'score-inf': entry['logprob'] = -float('inf')
    if case == 'score-huge': entry['logprob'] = -(10**400)
    if case == 'score-bytes': entry['bytes'] = [256]
    if case == 'score-byte-bool': entry['bytes'] = [True]
    if case == 'score-top-type': entry['top_logprobs'] = 'not records'
    if case == 'score-top-missing': entry.pop('top_logprobs')
    if case in ('score-top-valid','score-top-bad','score-top-extra'):
        entry['top_logprobs'] = [{'token':'hello','logprob':-.5,'bytes':None}]
    if case == 'score-top-bad': entry['top_logprobs'][0]['logprob'] = True
    if case == 'score-top-extra': entry['top_logprobs'] *= 2
    if case == 'score-too-many': content *= 3
    if case == 'score-optional-bytes': entry.pop('bytes')
    choice['logprobs'] = {'content':content}

chunks = [{'choices': [choice]}]
if case.startswith('error-'):
    chunks[0]['error'] = {'empty':{}, 'false':False, 'zero':0, 'list':[], 'text':'', 'null':None}[case[6:]]
if case.startswith('raw-'):
    choice.pop('delta',None);choice['text']='hello'
if case == 'bad-text': choice['delta']['content'] = ['hello']
if case == 'bad-delta': choice['delta'] = []
if case == 'multiple-choices': chunks[0]['choices'].append({'delta':{'content':'other'}})
finish = {'choices':[{'delta':{},'finish_reason':'stop'}]}
if case == 'invalid-finish': finish['choices'][0]['finish_reason'] = True
if case in ('terminal-output','raw-valid'):
    finish['choices'][0]['text' if case.startswith('raw-') else 'delta'] = 'end' if case.startswith('raw-') else {'content':'end'}
if case not in ('no-finish','raw-no-finish'): chunks.append(finish)
if case == 'duplicate-finish': chunks.append(finish)
if case == 'late-output': chunks.append({'choices':[{'delta':{'content':'late'}}]})
if case == 'late-reasoning': chunks.append({'choices':[{'delta':{'reasoning_content':'late'}}]})
if case == 'late-tool': chunks.append({'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'{}'}}]}}]})
if case != 'no-usage':
    tokens = {'short': 1, 'float': 2.0, 'bool': True}.get(case, 2)
    chunks.append({'choices': [], 'usage': {'completion_tokens': tokens}})
stream = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
if case == 'duplicate-json': stream = 'data: {"choices":[],"choices":[{"delta":{"content":"hello"}}]}\n\n' + stream
if case != 'no-done': stream += 'data: [DONE]\n\n'
args = ['compare', 'http://unused.invalid/v1', '--engine', 'dllm', '--max-tokens', '2']
if case.startswith('raw-'): args += ['--raw']
if case in ('no-scores','scored') or case.startswith('score-'):
    args += ['--top-logprobs', '1' if case.startswith('score-top-') else '0']
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
                         'error-empty','error-false','error-zero','error-list','error-text',
                         'no-finish','raw-no-finish','invalid-finish','duplicate-finish','late-output',
                         'bad-text','bad-delta','multiple-choices','duplicate-json','late-reasoning','late-tool'):
                with self.subTest(optimized=optimized, case=case):
                    result = self.run_case(case, optimized)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('ValueError', result.stderr)

    def test_malformed_scores_never_publish_comparison(self):
        cases = ('missing','string','entry','token','bool','positive','nan','inf','huge',
                 'bytes','byte-bool','top-type','top-missing','top-bad','top-extra','too-many')
        for optimized in (False, True):
            for case in cases:
                with self.subTest(optimized=optimized,case=case):
                    result = self.run_case('score-' + case,optimized)
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(result.stdout,'')
                    self.assertIn('ValueError',result.stderr)

    def test_complete_evidence_is_accepted_in_both_modes(self):
        for optimized in (False, True):
            for case in ('valid', 'scored', 'error-null', 'terminal-output', 'raw-valid',
                         'score-top-valid', 'score-optional-bytes'):
                with self.subTest(optimized=optimized, case=case):
                    result = self.run_case(case, optimized)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(len(report['trials']), 6)
                    self.assertTrue(all(row['usage']['completion_tokens'] == 2 for row in report['trials']))
