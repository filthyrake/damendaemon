import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm.deepseek_reference_loader import load_reference_model
from scripts.run_deepseek_reference import main
from tests.test_deepseek_reference_loader import fixture
from tests.test_deepseek_generation import text_fixture


class ReferenceCLITests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.path=Path(temporary.name)
        self.text=text_fixture(self.path)
        fixture(self.path,vocab_size=self.text.vocab_size)

    def test_loaded_checkpoint_command_and_declared_reference_scope(self):
        def controlled(*args,**kwargs):
            model=load_reference_model(*args,**kwargs)
            model.head['head.weight'].zero_()
            for token,sign in (('a',1),('b',-1)):
                model.head['head.weight'][self.text.tokenizer.token_to_id(token)].fill_(sign*.01)
            return model
        output=io.StringIO()
        with patch('scripts.run_deepseek_reference.load_reference_model',side_effect=controlled),contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path),'--prompt','Hello','--max-new-tokens','2','--context','128']),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]['scope'],'cpu_reference')
        self.assertFalse(events[0]['official_model_qualified'])
        self.assertEqual(events[-1]['usage']['completion_tokens'],2)
        self.assertEqual(events[-1]['finish_reason'],'length')
        self.assertEqual(len(''.join(e.get('content','') for e in events)),2)

    def test_invalid_prompt_context_and_limits_reject_before_weight_loading(self):
        messages=self.path/'messages.json'
        messages.write_text(json.dumps([{'role':'tool','content':'unsupported'}]))
        with patch('scripts.run_deepseek_reference.load_reference_model',side_effect=AssertionError('weights opened')), \
                contextlib.redirect_stderr(io.StringIO()):
            for args in (['--prompt','Hello','--context','1'],['--messages',str(messages)],
                         ['--prompt','Hello','--max-new-tokens','4097'],['--prompt','Hello','--weight-mib','0']):
                with self.assertRaises(SystemExit) as error:main([str(self.path),*args])
                self.assertEqual(error.exception.code,2)

    def test_broken_output_closes_generation_iterator(self):
        model=load_reference_model(self.path)
        closed=[]
        def events(*args,**kwargs):
            try:
                yield {'content':'first'}
                yield {'content':'second'}
            finally:closed.append(True)
        calls=0
        def fail_output(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls == 2:raise BrokenPipeError('consumer left')
        with patch('scripts.run_deepseek_reference.load_reference_model',return_value=model), \
                patch('scripts.run_deepseek_reference.ReferenceGenerator.generate',side_effect=events), \
                patch('builtins.print',side_effect=fail_output):
            self.assertEqual(main([str(self.path),'--prompt','Hello','--max-new-tokens','2','--context','128']),0)
        self.assertEqual(closed,[True])

    def test_tool_command_returns_calls_and_invalid_schemas_preflight(self):
        from dllm.deepseek_text import EOS
        from dllm.deepseek_tools import render_tool_calls
        from tests.test_deepseek_generation import ScriptedModel
        tools=self.path/'tools.json'
        tools.write_text(json.dumps([{'type':'function','function':{'name':'f'}}]))
        calls=[{'type':'function','function':{'name':'f','arguments':'{}'}}]
        model=ScriptedModel(self.text,'\n\n'+render_tool_calls(calls)+EOS)
        output=io.StringIO()
        with patch('scripts.run_deepseek_reference.load_reference_model',return_value=model),contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path),'--prompt','Hello','--tools',str(tools),'--max-new-tokens','512']),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertIn({'tool_calls':calls},events)
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        self.assertTrue(model.states[0].closed)
        for invalid in (None,[],[{'type':'function','function':{'name':'bad','parameters':[]}}]):
            tools.write_text(json.dumps(invalid))
            with patch('scripts.run_deepseek_reference.load_reference_model',side_effect=AssertionError('weights opened')),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([str(self.path),'--prompt','Hello','--tools',str(tools)])
                self.assertEqual(error.exception.code,2)
