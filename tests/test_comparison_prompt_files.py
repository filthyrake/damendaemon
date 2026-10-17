"""Token-file validation stays entirely before workspace/network activity."""
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from scripts import compare_a2,compare_llama
from damendaemon import token_workload


class ComparisonPromptFileTests(unittest.TestCase):
    def test_invalid_files_reject_before_workspace_or_http(self):
        payloads=[None,b'[]',b'[true]',b'[-1]',b'[1.0]',b'[NaN]',b'{}',b'\xff',
                  b'['*2000+b'0'+b']'*2000]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prompt.json'
            for payload in payloads:
                if payload is not None:path.write_bytes(payload)
                for module,args in ((compare_a2,[]),(compare_llama,['http://unused.invalid','--engine','damendaemon'])):
                    with self.subTest(payload=payload and payload[:20],module=module.__name__), \
                            patch('sys.argv',['compare',*args,'--prompt-token-ids',str(path)]), \
                            patch.object(compare_a2.Path,'cwd',side_effect=AssertionError('workspace accessed')), \
                            patch.object(compare_llama,'urlopen',side_effect=AssertionError('network accessed')), \
                            redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:module.main()
                        self.assertEqual(error.exception.code,2)

    def test_a2_enforces_prompt_plus_output_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prompt.json'
            for count in (192,193):
                path.write_text('['+','.join(['1']*count)+']')
                with self.subTest(count=count),patch('sys.argv',['compare','--prompt-token-ids',str(path)]), \
                        patch.object(compare_a2.Path,'cwd',side_effect=RuntimeError('workspace boundary')), \
                        redirect_stderr(io.StringIO()):
                    if count==192:
                        with self.assertRaisesRegex(RuntimeError,'workspace boundary'):compare_a2.main()
                    else:
                        with self.assertRaises(SystemExit) as error:compare_a2.main()
                        self.assertEqual(error.exception.code,2)

    def test_loader_exact_byte_and_token_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prompt.json';path.write_bytes(b'[0,12]')
            self.assertEqual(token_workload.load_token_prompt(path,max_bytes=6,max_tokens=2),[0,12])
            for options in ({'max_bytes':5},{'max_tokens':1},{'max_bytes':True},{'max_tokens':0}):
                with self.subTest(options=options),self.assertRaises(ValueError):
                    token_workload.load_token_prompt(path,**options)
