import io
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_glm_flash_reference as cli
from tests.test_glm_flash_generation import text_fixture,snapshot


class FlashReferenceCliTests(unittest.TestCase):
    def test_actual_packed_snapshot_runs_and_reports_cpu_only_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);snapshot(path,text)
            load=cli.load_reference_model;models=[]
            def prepared(*args,**kwargs):
                model=load(*args,**kwargs);model.head.zero_();models.append(model);return model
            output=io.StringIO()
            with patch.object(cli,'load_reference_model',side_effect=prepared),redirect_stdout(output):
                self.assertEqual(cli.main([str(path),'--prompt','Hi','--max-new-tokens','2','--context','128']),0)
            events=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[0],dict(scope='cpu_reference',official_model_qualified=False,dtype='float32'))
            self.assertEqual(events[-1]['finish_reason'],'length')
            self.assertEqual(events[-1]['usage']['completion_tokens'],2)
            self.assertFalse(models[0].failed_cleanups)

    def test_invalid_context_history_or_component_budget_precedes_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);snapshot(path,text)
            messages=path/'messages.json';messages.write_text(json.dumps([dict(role='tool',content='bad')]))
            args=([str(path),'--prompt','Hi','--context','1'],
                  [str(path),'--prompt','Hi','--cache-mib','2049'],
                  [str(path),'--messages',str(messages)])
            with patch.object(cli,'load_reference_model',side_effect=AssertionError('weights loaded')),redirect_stderr(io.StringIO()):
                for argv in args:
                    with self.assertRaises(SystemExit) as error:cli.main(argv)
                    self.assertEqual(error.exception.code,2)
