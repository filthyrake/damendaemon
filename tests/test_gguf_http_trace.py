from types import SimpleNamespace
import unittest

from tokenizers import AddedToken, Tokenizer, models
from scripts.verify_gguf_runtime import decoded_trace


class HttpTraceTests(unittest.TestCase):
    def test_eos_counts_as_generated_and_uses_the_sources_special_token_policy(self):
        tokenizer = Tokenizer(models.WordLevel({'hello': 0, 'end': 1, 'after': 2}))
        model = SimpleNamespace(tokenizer=tokenizer, eos={1})
        self.assertEqual(decoded_trace(model, [0, 1, 2]), dict(text='hello end', completion_tokens=2))
        tokenizer.add_special_tokens([AddedToken('end', special=True)])
        self.assertEqual(decoded_trace(model, [0, 1, 2]), dict(text='hello', completion_tokens=2))
        self.assertEqual(decoded_trace(model, [1, 2]), dict(text='', completion_tokens=1))
        self.assertEqual(decoded_trace(model, [0, 2]), dict(text='hello after', completion_tokens=2))


if __name__ == '__main__':
    unittest.main()
