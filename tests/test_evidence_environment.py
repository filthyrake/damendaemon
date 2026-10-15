"""Report writers must exclude credentials without importing hardware scripts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from dllm.evidence_environment import report_environment

SCRIPTS = ('compare_mps.py', 'compare_a2.py', 'verify_t5_service.py',
           'verify_prefix_service.py', 'profile_paged.py', 'verify_tools_service.py',
           'verify_logprobs.py', 'benchmark_chunked.py', 'compare_paged.py')


def fixture():
    return {'DLLM_DEVICE': 'cuda', 'DLLM_KERNELS': 'triton', 'DLLM_KV_CACHE_TOKENS': '4096',
            **{name: 'synthetic-sensitive-value' for name in (
                'DLLM_API_KEY', 'DLLM_ADMIN_API_KEY', 'DLLM_CLIENT_KEYS',
                'DLLM_REMOTE_QUOTA_OWNER_KEY', 'DLLM_REMOTE_QUOTA_TLS_PASSWORD',
                'DLLM_REMOTE_QUOTA_CONFIG', 'DLLM_CLIENT_RATE_DB', 'DLLM_WEIGHT_CACHE',
                'DLLM_AWQ_MARLIN_LIBRARY', 'DLLM_FUTURE_CREDENTIAL', 'OTHER_SECRET')}}


class EvidenceEnvironmentTests(unittest.TestCase):
    def test_every_report_expression_excludes_sensitive_and_unknown_values(self):
        environment = fixture()
        expected = {key: environment[key] for key in
                    ('DLLM_DEVICE', 'DLLM_KERNELS', 'DLLM_KV_CACHE_TOKENS')}
        for name in SCRIPTS:
            with self.subTest(script=name):
                path = Path(__file__).resolve().parents[1] / 'scripts' / name
                tree = ast.parse(path.read_text())
                values = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                          for key, value in zip(node.keys, node.values)
                          if isinstance(key, ast.Constant) and key.value == 'runtime_environment']
                self.assertEqual(len(values), 1)
                expression = ast.Expression(values[0])
                report = eval(compile(expression, str(path), 'eval'),
                              {'env': environment, 'os': SimpleNamespace(environ=environment),
                               'report_environment': report_environment})
                self.assertEqual(report, expected)

    def test_filter_does_not_mutate_input_or_expose_future_keys(self):
        source = fixture()
        original = source.copy()
        result = report_environment(source)
        self.assertEqual(source, original)
        result['DLLM_DEVICE'] = 'mps'
        self.assertEqual(source['DLLM_DEVICE'], 'cuda')
        self.assertEqual(report_environment({'DLLM_NEW_SECRET': 'synthetic-sensitive-value'}), {})
        self.assertEqual(report_environment({}), {})
