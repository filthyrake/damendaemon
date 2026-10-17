"""Report writers must exclude credentials without importing hardware scripts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from damendaemon.evidence_environment import report_environment

SCRIPTS = ('compare_mps.py', 'compare_a2.py', 'verify_t5_service.py',
           'verify_prefix_service.py', 'profile_paged.py', 'verify_tools_service.py',
           'verify_logprobs.py', 'benchmark_chunked.py', 'compare_paged.py')


def fixture():
    return {'DAMENDAEMON_DEVICE': 'cuda', 'DAMENDAEMON_KERNELS': 'triton', 'DAMENDAEMON_KV_CACHE_TOKENS': '4096',
            **{name: 'synthetic-sensitive-value' for name in (
                'DAMENDAEMON_API_KEY', 'DAMENDAEMON_ADMIN_API_KEY', 'DAMENDAEMON_CLIENT_KEYS',
                'DAMENDAEMON_REMOTE_QUOTA_OWNER_KEY', 'DAMENDAEMON_REMOTE_QUOTA_TLS_PASSWORD',
                'DAMENDAEMON_REMOTE_QUOTA_CONFIG', 'DAMENDAEMON_CLIENT_RATE_DB', 'DAMENDAEMON_WEIGHT_CACHE',
                'DAMENDAEMON_AWQ_MARLIN_LIBRARY', 'DAMENDAEMON_FUTURE_CREDENTIAL', 'OTHER_SECRET')}}


class EvidenceEnvironmentTests(unittest.TestCase):
    def test_every_report_expression_excludes_sensitive_and_unknown_values(self):
        environment = fixture()
        expected = {key: environment[key] for key in
                    ('DAMENDAEMON_DEVICE', 'DAMENDAEMON_KERNELS', 'DAMENDAEMON_KV_CACHE_TOKENS')}
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
        result['DAMENDAEMON_DEVICE'] = 'mps'
        self.assertEqual(source['DAMENDAEMON_DEVICE'], 'cuda')
        self.assertEqual(report_environment({'DAMENDAEMON_NEW_SECRET': 'synthetic-sensitive-value'}), {})
        self.assertEqual(report_environment({}), {})
