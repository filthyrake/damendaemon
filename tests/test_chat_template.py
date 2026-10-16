import tempfile
import unittest
from pathlib import Path

from jinja2.exceptions import SecurityError
from dllm.chat_template import compile_template, validate_override, read_override, MAX_TEMPLATE_BYTES
from dllm.management import command


class ChatTemplateTests(unittest.TestCase):
    def test_validation_before_registration_and_admission(self):
        text = '{{ messages | tojson }}'
        self.assertEqual(command('register', 'a', '/model', chat_template=text)['chat_template'], text)
        for value in ('', ' ', [], True, '{{', '{% for x in %}', 'x\x00', '\ud800', 'x'*(MAX_TEMPLATE_BYTES+1)):
            with self.subTest(value=repr(value)[:30]), self.assertRaises(ValueError):
                command('register', 'a', '/model', chat_template=value)
        for action in ('status', 'load', 'unload'):
            with self.assertRaises(ValueError):
                command(action, None if action=='status' else 'a', chat_template=text)
        for kind in ('mamba', 'mamba2', 't5', 'bert'):
            with self.assertRaises(ValueError):
                validate_override(text, {'model_type':kind, 'task':'embedding' if kind=='bert' else None})

    def test_sandbox_and_reference_json_semantics(self):
        template = compile_template('{{ value | tojson }}')
        self.assertEqual(template.render(value={'z':'<é>', 'a':1}), '{"z": "<é>", "a": 1}')
        with self.assertRaises(SecurityError):
            compile_template("{{ ''.__class__.__mro__ }}").render()
        with self.assertRaises(SecurityError):
            compile_template("{{ values.append(1) }}").render(values=[])

    def test_file_is_bounded_utf8_and_copied_as_text(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'template.jinja'
            file.write_text('{{ messages }}', encoding='utf-8')
            self.assertEqual(read_override(file), '{{ messages }}')
            file.write_bytes(b'x'*(MAX_TEMPLATE_BYTES+1))
            with self.assertRaises(ValueError):read_override(file)
            file.write_bytes(b'\xff')
            with self.assertRaises(ValueError):read_override(file)

    def test_checkpoint_structured_tokens_render_as_token_text(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from dllm.qwen2 import ModelWeights
        tokenizer = Tokenizer(models.WordLevel({'[UNK]':0,'BOS':1,'hello':2,'EOS':3},unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        model = ModelWeights.__new__(ModelWeights)
        model.config = {'model_type':'qwen2','vocab_size':4}
        template = '{{ bos_token }} {{ messages[0]["content"] }} {{ eos_token }}'
        for structured in (False,True):
            with self.subTest(structured=structured):
                metadata = {'bos_token':'BOS','eos_token':'EOS'}
                if structured:
                    metadata = {key:{'content':value,'lstrip':False,'special':True} for key,value in metadata.items()}
                model.configure_tokenizer(tokenizer,template,metadata)
                self.assertEqual(model.template.render(messages=[{'content':'hello'}],**model.special_tokens),'BOS hello EOS')
                self.assertEqual(model.chat_tokens([{'role':'user','content':'hello'}]),[1,2,3])

    def test_render_error_normalization_is_narrow_and_preserves_cause(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from jinja2 import UndefinedError
        from dllm.qwen2 import ModelWeights
        for error in (UndefinedError('missing field'),SecurityError('blocked'),RecursionError('nested')):
            with self.subTest(error=type(error).__name__):
                model=SimpleNamespace(config={'model_type':'qwen2'},special_tokens={},
                    template=SimpleNamespace(render=Mock(side_effect=error)),tokenizer=Mock())
                with self.assertRaisesRegex(ValueError,'chat template rendering failed') as raised:
                    ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}])
                self.assertIs(raised.exception.__cause__,error)
                model.tokenizer.encode.assert_not_called()
        failure=RuntimeError('backend failure')
        model.template.render.side_effect=failure
        with self.assertRaises(RuntimeError) as raised:
            ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}])
        self.assertIs(raised.exception,failure)
