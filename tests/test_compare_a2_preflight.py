"""No workspace reads, sockets, child processes, or GPU access."""
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from scripts import compare_a2


class ComparisonPreflightTests(unittest.TestCase):
    def test_invalid_options_fail_before_external_work(self):
        cases=[['--startup-timeout',value] for value in ('0','-1','86401')]
        cases += [['--context',value] for value in ('0','-1','1048577')]
        cases += [['--host-gib',value] for value in ('0','-1','nan','inf','1e308','1e-320')]
        cases += [['--max-tokens',value] for value in ('0','1','256','257')]
        cases += [['--raw','--top-logprobs','0'],['--prompt-token-ids','unread.json','--top-logprobs','1']]
        for args in cases:
            with self.subTest(args=args),patch('sys.argv',['compare',*args]),redirect_stderr(io.StringIO()), \
                    patch.object(compare_a2.Path,'cwd',side_effect=AssertionError('workspace accessed')) as cwd, \
                    patch.object(compare_a2.socket,'socket',side_effect=AssertionError('socket created')) as socket, \
                    patch.object(compare_a2.subprocess,'Popen',side_effect=AssertionError('process started')) as process:
                with self.assertRaises(SystemExit) as error:compare_a2.main()
                self.assertEqual(error.exception.code,2)
                cwd.assert_not_called();socket.assert_not_called();process.assert_not_called()

    def test_valid_options_reach_workspace_boundary(self):
        for args in ([],['--context','1048576','--max-tokens','2','--startup-timeout','86400','--host-gib','0.000001'],
                     ['--raw'],['--prompt-token-ids','unread.json'],['--top-logprobs','0']):
            with self.subTest(args=args),patch('sys.argv',['compare',*args]), \
                    patch.object(compare_a2.Path,'cwd',side_effect=RuntimeError('workspace boundary')):
                with self.assertRaisesRegex(RuntimeError,'workspace boundary'):compare_a2.main()
