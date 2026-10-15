"""Run the benchmark CLI with no actual server, tokenizer file or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io, json, sys
from types import SimpleNamespace
from unittest.mock import patch
from scripts import benchmark_chunked as benchmark
case = sys.argv[1]
processes = []
chunk = [0]
class Tokenizer:
    def no_padding(self): pass
    def no_truncation(self): pass
    def encode(self, text, **kwargs):
        count = (10 if case == 'short-corpus' else 1536) if text.startswith('The blue bird') else 4
        return SimpleNamespace(ids=[1]*count)
class Process:
    terminated = False
    def poll(self): return None
    def terminate(self): self.terminated = True
    def wait(self, *args): return 0
    def kill(self): raise AssertionError('not needed')
def launch(command, stdout, **kwargs):
    chunk[0] = int(command[command.index('--prefill-chunk-size')+1])
    stdout.write('dllm listening on http://unused.invalid\n'); stdout.flush()
    process=Process(); processes.append(process); return process
def response(request, **kwargs):
    body=json.loads(request.data)
    usage=dict(prompt_tokens=len(body['prompt']), completion_tokens=body['max_tokens'])
    if not body['stream']:
        if case == 'long-usage' and len(body['prompt']) > 4: usage['completion_tokens']=1
        text='changed' if case == 'diverge' and chunk[0] else 'long'
        return io.BytesIO(json.dumps(dict(choices=[dict(text=text)],usage=usage)).encode())
    chunks=[dict(choices=[dict(text=letter)]) for letter in 'abcde']
    if case != 'no-usage' and body.get('stream_options', {}).get('include_usage'):
        if case == 'short-usage': usage['completion_tokens']=1
        chunks.append(dict(choices=[],usage=usage))
    wire=''.join('data: '+json.dumps(value)+'\n\n' for value in chunks)
    if case != 'no-done': wire+='data: [DONE]\n\n'
    return io.BytesIO(wire.encode())
args=['benchmark','unused-model','--rounds','0' if case=='zero-rounds' else '1']
try:
    with patch('sys.argv',args), patch.object(benchmark.Tokenizer,'from_file',return_value=Tokenizer()), \
            patch.object(benchmark.subprocess,'Popen',side_effect=launch), patch.object(benchmark,'urlopen',side_effect=response):
        benchmark.main()
finally:
    print('processes='+str(len(processes))+' cleaned='+str(all(p.terminated for p in processes)),file=sys.stderr)
'''


class ChunkedBenchmarkTests(unittest.TestCase):
    def run_case(self, case, optimized):
        return subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', HARNESS, case],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)

    def test_incomplete_work_never_publishes_metrics_and_always_cleans_up(self):
        for optimized in (False, True):
            for case in ('short-corpus','zero-rounds','no-done','no-usage','short-usage','long-usage','diverge'):
                with self.subTest(optimized=optimized, case=case):
                    result=self.run_case(case,optimized)
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(result.stdout,'')
                    self.assertIn('cleaned=True',result.stderr)
                    if case in ('short-corpus','zero-rounds'):
                        self.assertIn('processes=0',result.stderr)

    def test_valid_workload_runs_each_chunk_size_in_both_modes(self):
        for optimized in (False,True):
            result=self.run_case('valid',optimized)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual([row['chunk_size'] for row in report['results']],[0,128,256])
            self.assertTrue(report['same_greedy_outputs'])
            self.assertIn('processes=3 cleaned=True',result.stderr)
