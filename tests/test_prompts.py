from types import SimpleNamespace
import unittest
from dllm.prompts import validate_raw, encode
from dllm.qwen2 import ModelWeights


class PromptTests(unittest.TestCase):
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

    def test_dispatch_rejects_mixed_modes(self):
        model = SimpleNamespace(raw_tokens=lambda value: value)
        self.assertEqual(encode(model, None, [1]), [1])
        for messages, thinking in (([], None), (None, False)):
            with self.assertRaises(ValueError):
                encode(model, messages, [1], thinking)
        with self.assertRaisesRegex(ValueError, 'no chat template'):
            ModelWeights.chat_tokens(SimpleNamespace(template=None), [])
