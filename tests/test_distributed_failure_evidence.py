"""Aggregate evidence freshness, without launching any distributed workers."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import distributed_job, verify_distributed_failure as runner
sys.modules['distributed_job'] = distributed_job
case = sys.argv[1]
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory)/'result.json'
    coordinator = dict(passed=True, requests=[{}], run_id='previous')
    peer = dict(injected=True, operation='decode', rows=4, run_id='previous')
    paths = [output.with_name('result-coordinator.json'), output.with_name('result-worker.json')]
    def save():
        for path, report in zip(paths, (coordinator, peer)):
            path.write_text(json.dumps(report))
    save()
    def job(command, **kwargs):
        run_id = command[command.index('--run-id')+1] if '--run-id' in command else 'previous'
        if case != 'stale': coordinator['run_id'] = run_id
        if case not in ('stale', 'mixed'): peer['run_id'] = run_id
        if case == 'failed': coordinator['passed'] = False
        if case == 'not-injected': peer['injected'] = False
        if case == 'wrong-operation': peer['operation'] = 'prefill'
        if case == 'wrong-rows': peer['rows'] = 1
        if case == 'prefill': peer.update(operation='prefill', rows=1)
        save()
        return SimpleNamespace(returncode=0)
    argv = ['runner', 'unused-model', '--output', str(output)]
    if case == 'prefill': argv += ['--operation', 'prefill']
    with patch.object(distributed_job, 'run_job', side_effect=job), patch('sys.argv', argv):
        try: runner.main()
        except Exception:
            if output.exists(): raise RuntimeError('invalid aggregate was published')
            raise
    aggregate = json.loads(output.read_text())
    assert aggregate['coordinator']['run_id'] == aggregate['worker']['run_id']
'''


class DistributedFailureEvidenceTests(unittest.TestCase):
    def run_case(self, case, optimized):
        return subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', HARNESS, case],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)

    def test_stale_mixed_and_failed_evidence_cannot_pass(self):
        for optimized in (False, True):
            for case in ('stale', 'mixed', 'failed', 'not-injected', 'wrong-operation', 'wrong-rows'):
                with self.subTest(case=case, optimized=optimized):
                    result = self.run_case(case, optimized)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('ValueError', result.stderr)

    def test_current_matching_evidence_passes(self):
        for optimized in (False, True):
            for case in ('valid', 'prefill'):
                with self.subTest(optimized=optimized, case=case):
                    result = self.run_case(case, optimized)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(json.loads(result.stdout)['passed'])
