from types import SimpleNamespace
import unittest
from dllm.prompts import validate_raw, encode
from dllm.qwen2 import ModelWeights


class PromptTests(unittest.TestCase):
    def test_invalid_unicode_rejected_before_raw_runtime_activation(self):
        from unittest.mock import Mock
        from dllm.runtime import Runtime
        from dllm.stopping import stop_sequences
        from dllm.tool_calls import validate_messages
        runtime = Runtime.__new__(Runtime)
        runtime.context = 1024
        runtime.session = Mock(side_effect=AssertionError('must not activate model'))
        for text in ('\ud800', '\udfff', 'prefix\ud800suffix'):
            for call in (lambda: validate_raw(text),
                         lambda: stop_sequences(text),
                         lambda: validate_messages([{'role':'user', 'content':text}]),
                         lambda: validate_messages([{'role':'assistant', 'content':'ok', 'reasoning_content':text}]),
                         lambda: next(runtime.generate('model', None, raw_prompt=text)),
                         lambda: runtime.score('model', text, [1, 2])):
                with self.subTest(text=ascii(text)), self.assertRaises(ValueError):
                    call()
        runtime.session.assert_not_called()
        text = 'café 世界 😀'
        self.assertEqual(validate_raw(text), text)
        self.assertEqual(stop_sequences(text), (text,))
        self.assertEqual(validate_messages([{'role':'user', 'content':text}]), [{'role':'user', 'content':text}])
        history = [{'role':'assistant', 'content':None, 'tool_calls':[{
            'id':'call_one', 'type':'function', 'function':{'name':'weather', 'arguments':'{}'}}]},
            {'role':'tool', 'tool_call_id':'call_one', 'content':text}]
        self.assertEqual(validate_messages(history), history)
        history[1]['content'] = '\ud800'
        with self.assertRaises(ValueError):
            validate_messages(history)

    def test_raw_ids_are_copied_and_checked(self):
        tokens = [1, 2]
        self.assertIsNot(validate_raw(tokens), tokens)
        for value in ([], [True], [-1], [1.5], ['text'], [[1]], None, {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_raw(value)
        model = SimpleNamespace(config={'vocab_size': 10, 'multimodal_token_ids': [8]})
        self.assertEqual(ModelWeights.raw_tokens(model, tokens), tokens)
        for value in ([10], [8]):
            with self.assertRaises(ValueError):
                ModelWeights.raw_tokens(model, value)

    def test_raw_encoding_uses_tokenizer_special_tokens(self):
        calls = []
        def tokenize(text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(ids=[1, 2])
        model = SimpleNamespace(config={'vocab_size': 10}, tokenizer=SimpleNamespace(encode=tokenize))
        self.assertEqual(ModelWeights.raw_tokens(model, 'hello'), [1, 2])
        self.assertEqual(calls, [('hello', {'add_special_tokens': True})])

    def test_chat_tokens_are_nonempty_and_inside_model_vocabulary(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from dllm.chat_template import compile_template
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'ok': 1, 'extra': 2}, unk_token='[UNK]'))
        model = SimpleNamespace(config={'model_type': 'qwen2', 'vocab_size': 2},
            tokenizer=tokenizer, template=compile_template('{{ messages[0].content }}'),
            special_tokens={}, supports_tool_calls=False)
        messages = [{'role': 'user', 'content': 'ok'}]
        self.assertEqual(ModelWeights.chat_tokens(model, messages), [1])
        messages[0]['content'] = 'extra'
        for template in ('{{ messages[0].content }}', '{{ "" }}'):
            with self.subTest(template=template):
                model.template = compile_template(template)
                with self.assertRaisesRegex(ValueError, 'vocabulary'):
                    ModelWeights.chat_tokens(model, messages)
        model.template = compile_template('{{ messages[0].content }}')
        messages[0]['content'] = 'ok'
        model.config['multimodal_token_ids'] = [1]
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            ModelWeights.chat_tokens(model, messages)

    def test_dispatch_rejects_mixed_modes(self):
        model = SimpleNamespace(raw_tokens=lambda value: value)
        self.assertEqual(encode(model, None, [1]), [1])
        for messages, thinking in (([], None), (None, False)):
            with self.assertRaises(ValueError):
                encode(model, messages, [1], thinking)
        with self.assertRaisesRegex(ValueError, 'no chat template'):
            ModelWeights.chat_tokens(SimpleNamespace(template=None), [])
