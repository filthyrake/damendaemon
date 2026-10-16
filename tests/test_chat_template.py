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
                    template=SimpleNamespace(generate=Mock(side_effect=error)),tokenizer=Mock())
                with self.assertRaisesRegex(ValueError,'chat template rendering failed') as raised:
                    ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}])
                self.assertIs(raised.exception.__cause__,error)
                model.tokenizer.encode.assert_not_called()
        failure=RuntimeError('backend failure')
        model.template.generate.side_effect=failure
        with self.assertRaises(RuntimeError) as raised:
            ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}])
        self.assertIs(raised.exception,failure)

    def test_actual_template_expression_failures_before_tokenization(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from dllm.qwen2 import ModelWeights
        from dllm.cpu_reference_chat import CpuReferenceChat
        cases = (('{{ messages[0].content + 1 }}', TypeError),
                 ('{{ range(3) | tojson }}', TypeError),
                 ('{{ 1 / (messages | length - 1) }}', ZeroDivisionError))
        for source, error_type in cases:
            for implementation in (ModelWeights.chat_tokens, CpuReferenceChat.chat_tokens):
                with self.subTest(source=source, implementation=implementation.__qualname__):
                    model = SimpleNamespace(config={'model_type':'qwen2','vocab_size':4},
                        template=compile_template(source), special_tokens={}, tokenizer=Mock())
                    with self.assertRaisesRegex(ValueError, 'template rendering failed') as raised:
                        implementation(model, [{'role':'user','content':'hello'}])
                    self.assertIsInstance(raised.exception.__cause__, error_type)
                    model.tokenizer.encode.assert_not_called()

    def test_expanded_prompt_bytes_are_bounded_before_tokenization(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from dllm.qwen2 import ModelWeights
        template=compile_template('{% for i in range(10) %}{{ messages[0].content }}{% endfor %}')
        model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':2},special_tokens={},
            template=template,tokenizer=SimpleNamespace(encode=Mock(return_value=SimpleNamespace(ids=[1]))))
        with patch('dllm.qwen2.MAX_RENDERED_CHAT_BYTES',32):
            with self.assertRaisesRegex(ValueError,'rendered prompt'):
                ModelWeights.chat_tokens(model,[{'role':'user','content':'éé'}])
        model.tokenizer.encode.assert_not_called()

    def test_render_budget_stops_and_closes_generator_and_counts_tool_prefix(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from dllm.qwen2 import ModelWeights
        consumed=[];closed=[]
        def generate(**options):
            try:
                for index in range(100):
                    consumed.append(index)
                    yield 'abcd'
            finally:closed.append(True)
        model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':2},special_tokens={},
            template=SimpleNamespace(generate=generate),supports_tool_calls=True,
            tokenizer=SimpleNamespace(encode=Mock(return_value=SimpleNamespace(ids=[1]))))
        with patch('dllm.qwen2.MAX_RENDERED_CHAT_BYTES',8):
            with self.assertRaisesRegex(ValueError,'rendered prompt'):
                ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}])
        self.assertEqual(consumed,[0,1,2]);self.assertEqual(closed,[True])
        model.tokenizer.encode.assert_not_called()
        model.template=compile_template('éééé')
        with patch('dllm.qwen2.MAX_RENDERED_CHAT_BYTES',8):
            self.assertEqual(ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}]),[1])
        self.assertEqual(model.tokenizer.encode.call_args.args[0],'éééé')
        tools=[{'type':'function','function':{'name':'weather','parameters':{'type':'object','properties':{}}}}]
        model.tokenizer.encode.reset_mock()
        budget=8+len('<tool_call>\n')
        with patch('dllm.qwen2.MAX_RENDERED_CHAT_BYTES',budget):
            self.assertEqual(ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}],tools=tools,tool_json=True),[1])
        self.assertEqual(model.tokenizer.encode.call_args.args[0],'éééé<tool_call>\n')
        model.tokenizer.encode.reset_mock()
        with patch('dllm.qwen2.MAX_RENDERED_CHAT_BYTES',budget-1):
            with self.assertRaisesRegex(ValueError,'rendered prompt'):
                ModelWeights.chat_tokens(model,[{'role':'user','content':'hello'}],tools=tools,tool_json=True)
        model.tokenizer.encode.assert_not_called()


class NamedTemplateCompatibilityTests(unittest.TestCase):
    def test_default_rendering_matches_local_transformers(self):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from dllm.tokenizer_metadata import checkpoint_template
        from dllm.qwen2 import ModelWeights
        template = "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
        messages = [{'role':'system','content':'Be brief'}, {'role':'user','content':'héllo'}]
        tokenizer = Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]'))
        for named in ({'tool_use':'wrong variant','default':template},
                      [{'name':'tool_use','template':'wrong variant'}, {'name':'default','template':template}]):
            with self.subTest(named=named):
                reference = PreTrainedTokenizerFast(tokenizer_object=tokenizer,chat_template=named)
                selected = checkpoint_template('.', {'chat_template':named})
                model = ModelWeights.__new__(ModelWeights)
                model.config = {'model_type':'qwen2','vocab_size':1}
                model.configure_tokenizer(tokenizer, selected, {})
                for generation in (False, True):
                    expected = reference.apply_chat_template(messages,tokenize=False,add_generation_prompt=generation)
                    self.assertEqual(model.template.render(messages=messages,add_generation_prompt=generation,**model.special_tokens),expected)
