"""Exercise real failure-client acceptance with in-memory HTTP responses."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HARNESS = r'''
import io, json, sys
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from scripts.verify_distributed_failure import failure_client
case = sys.argv[1]
report = {'passed': False}
status = dict(failed=True, closed=True, active=False, completed=0, queued=0)
if case == 'active': status['active'] = True
if case == 'completed': status['completed'] = 1
if case == 'queued': status['queued'] = 1
if case == 'not-failed': status['failed'] = False
runtime = NS(closed=NS(wait=Mock(return_value=case != 'not-closed')),
             manage=Mock(return_value=status), pair=NS(state=NS(failed=case != 'transport-live')),
             close=Mock())
responses = []
def http(request, **kwargs):
    data = json.loads(request.data) if request.data else None
    if data and data.get('max_tokens') == 32:
        if data['stream']:
            chunks = [{'error': {'message': 'injected'}}]
            if case == 'no-error': chunks = [{'choices': []}]
            if case == 'finish': chunks.append({'choices': [{'finish_reason': 'stop'}]})
            if case == 'usage': chunks.append({'usage': {'completion_tokens': 1}})
            if case == 'empty-usage': chunks.append({'usage': {}})
            stream = ''.join('data: '+json.dumps(item)+'\n\n' for item in chunks)
            if case == 'done': stream += 'data: [DONE]\n\n'
            return io.BytesIO(stream.encode())
        code = 400 if case == 'wrong-status' else 500
        payload = {'error': {'message': 'injected'}}
        if case == 'choices': payload['choices'] = []
    else:
        code = 400 if case == 'closed-status' else 503
        payload = {'error': {'message': 'closed'}}
    response = io.BytesIO(b'{' if case == 'malformed-json' else json.dumps(payload).encode())
    responses.append(response)
    raise HTTPError(request.full_url, code, 'fixture', {}, response)
with patch('scripts.verify_distributed_failure.urlopen', side_effect=http):
    failure_client(runtime, NS(stream=True, operation='decode'), 'http://unused.invalid', report)
print(json.dumps(dict(report=report, closes=runtime.close.call_count,
                      waits=runtime.closed.wait.call_count, responses_closed=all(r.closed for r in responses))))
'''


class DistributedFailureClientTests(unittest.TestCase):
    def run_case(self, case, optimized):
        result = subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', HARNESS, case],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_invalid_failure_evidence_is_rejected(self):
        for optimized in (False, True):
            for case in ('no-error', 'finish', 'usage', 'done', 'wrong-status', 'choices',
                         'closed-status', 'not-closed', 'active', 'completed', 'queued',
                         'not-failed', 'transport-live', 'empty-usage', 'malformed-json'):
                with self.subTest(case=case, optimized=optimized):
                    result = self.run_case(case, optimized)
                    self.assertFalse(result['report']['passed'])
                    self.assertIn('error', result['report'])
                    self.assertEqual(result['closes'], 1)
                    self.assertTrue(result['responses_closed'])

    def test_valid_failure_evidence_waits_for_cleanup_in_both_modes(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                result = self.run_case('valid', optimized)
                self.assertTrue(result['report']['passed'], result)
                self.assertEqual(result['closes'], 1)
                self.assertEqual(result['waits'], 1)
