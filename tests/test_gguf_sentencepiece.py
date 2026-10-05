import importlib.util
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from tokenizers import Tokenizer

HAS_GGUF = importlib.util.find_spec('gguf') is not None
if HAS_GGUF:
    from dllm.gguf_model import tokenizer


def metadata():
    pieces = ['<unk>', '<s>', '</s>'] + [f'<0x{i:02X}>' for i in range(256)]
    pieces += ['▁', 'a', 'b', 'ab', '▁a', '▁ab', '<user>']
    return {'general.architecture': 'llama', 'tokenizer.ggml.model': 'llama',
            'tokenizer.ggml.tokens': pieces,
            'tokenizer.ggml.token_type': [2, 3, 3] + [6] * 256 + [1] * 6 + [4],
            'tokenizer.ggml.scores': [0.] * 259 + [-4., -1., -2., -3., -5., -6., 0.],
            'tokenizer.ggml.bos_token_id': 1, 'tokenizer.ggml.eos_token_id': 2,
            'tokenizer.ggml.unknown_token_id': 0}


@unittest.skipUnless(HAS_GGUF, 'optional gguf dependency is not installed')
class SentencePieceGgufTests(unittest.TestCase):
    def test_score_merges_whitespace_and_json_roundtrip(self):
        result, special, eos = tokenizer(metadata())
        self.assertEqual(special, {'bos_token': '<s>', 'eos_token': '</s>'})
        self.assertEqual(eos, frozenset({2}))
        self.assertEqual(result.encode('ab').ids, [1, 264])
        self.assertEqual(result.encode(' ab', add_special_tokens=False).ids, [259, 264])
        self.assertEqual(result.encode('', add_special_tokens=False).ids, [])
        self.assertEqual(result.encode('').ids, [1])
        restored = Tokenizer.from_str(result.to_str())
        for text in ('ab', ' ab', '  ab  ab', '\t ab\n', 'café e\u0301 中文 😀', 'a\x00b'):
            with self.subTest(text=text):
                ids = restored.encode(text, add_special_tokens=False).ids
                self.assertEqual(ids, result.encode(text, add_special_tokens=False).ids)
                self.assertEqual(restored.decode(ids), text)

    def test_special_boundaries_and_user_defined_tokens(self):
        result, _, _ = tokenizer(metadata())
        self.assertEqual(result.encode('ab</s>ab', add_special_tokens=False).ids, [264, 2, 264])
        self.assertEqual(result.encode('<s>ab').ids, [1, 1, 264])
        self.assertEqual(result.encode('<unk>', add_special_tokens=False).ids, [0])
        self.assertEqual(result.encode('<user>', add_special_tokens=False).ids, [265])
        self.assertEqual(result.decode([265]), '<user>')
        self.assertEqual(result.decode([0, 1, 2]), '')
        self.assertEqual(result.decode([1, 264]), ' ab')
        self.assertEqual(result.decode_batch([[264], [1, 264], [0, 264]]), ['ab', ' ab', ' ab'])
        self.assertIsInstance(result.backend_tokenizer, Tokenizer)

    def test_generation_preserves_leading_space_without_changing_text_decode(self):
        from dllm.prompts import decode_output
        result, _, _ = tokenizer(metadata())
        self.assertEqual(result.decode([264]), 'ab')
        self.assertEqual(decode_output(SimpleNamespace(tokenizer=result), [264]), ' ab')
        self.assertEqual(decode_output(SimpleNamespace(tokenizer=result), [1, 264]), ' ab')
        plain = Tokenizer.from_str(result.to_str())
        self.assertEqual(decode_output(SimpleNamespace(tokenizer=plain), [264]), 'ab')

    def test_declared_unknown_can_be_a_control_token(self):
        source = metadata()
        source['tokenizer.ggml.token_type'][0] = 3
        result, _, _ = tokenizer(source)
        self.assertEqual(result.encode('<unk>ab', add_special_tokens=False).ids, [0, 264])
        self.assertEqual(result.decode([0, 264]), ' ab')

    def test_disabled_prefix_and_explicit_eos(self):
        source = metadata()
        source.update({'tokenizer.ggml.add_space_prefix': False,
                       'tokenizer.ggml.add_bos_token': False,
                       'tokenizer.ggml.add_eos_token': True})
        result, _, _ = tokenizer(source)
        self.assertEqual(result.encode('ab').ids, [262, 2])
        self.assertEqual(result.encode(' ab').ids, [264, 2])
        self.assertEqual(result.decode([264]), ' ab')

    def test_full_gguf_load_preserves_decode_policy_and_chat_template(self):
        import numpy as np
        from gguf import GGUFWriter
        from dllm.gguf_model import GgufWeights, tensor_groups
        source = metadata()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sentencepiece.gguf'
            writer = GGUFWriter(str(path), 'llama')
            for key, value in {'context_length': 128, 'embedding_length': 32,
                               'feed_forward_length': 64, 'block_count': 1,
                               'attention.head_count': 4, 'attention.head_count_kv': 2}.items():
                writer.add_uint32('llama.' + key, value)
            writer.add_float32('llama.attention.layer_norm_rms_epsilon', 1e-6)
            writer.add_tokenizer_model('llama')
            writer.add_token_list(source['tokenizer.ggml.tokens'])
            writer.add_token_types(source['tokenizer.ggml.token_type'])
            writer.add_token_scores(source['tokenizer.ggml.scores'])
            writer.add_bos_token_id(1); writer.add_eos_token_id(2); writer.add_unk_token_id(0)
            writer.add_chat_template("{{ bos_token }}{{ messages[0]['content'] }}")
            config = dict(model_type='llama', hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, vocab_size=len(source['tokenizer.ggml.tokens']))
            for _, group in tensor_groups(config, {'output.weight': True}):
                for name, shape in group:
                    writer.add_tensor(name, np.ones(shape, dtype=np.float16))
            writer.write_header_to_file(); writer.write_kv_data_to_file()
            writer.write_tensors_to_file(); writer.close()
            model = GgufWeights(path, 64, 'none')
            self.assertEqual(model.tokenizer.decode([1, 264]), ' ab')
            self.assertEqual(model.chat_tokens([{'role': 'user', 'content': 'ab'}]), [1, 264])

    def test_invalid_vocab_and_unsupported_normalization_are_rejected(self):
        mutations = [
            lambda m: m.pop('tokenizer.ggml.scores'),
            lambda m: m['tokenizer.ggml.scores'].append(0.),
            lambda m: m['tokenizer.ggml.scores'].__setitem__(260, math.nan),
            lambda m: m['tokenizer.ggml.scores'].__setitem__(263, -3.),
            lambda m: m['tokenizer.ggml.tokens'].__setitem__(260, 'b'),
            lambda m: m['tokenizer.ggml.token_type'].__setitem__(3, 1),
            lambda m: m.update({'tokenizer.ggml.unknown_token_id': 1}),
            lambda m: m.update({'tokenizer.ggml.add_space_prefix': 1}),
            lambda m: m.update({'tokenizer.ggml.remove_extra_whitespaces': True}),
            lambda m: m.update({'tokenizer.ggml.pre': 'uncovered'}),
            lambda m: m.update({'tokenizer.ggml.precompiled_charsmap': [1, 2]}),
            lambda m: m.update({'tokenizer.ggml.normalizer.lowercase': True}),
            lambda m: m.update({'general.architecture': 'qwen2'}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                source = metadata()
                mutate(source)
                with self.assertRaises(ValueError):tokenizer(source)


if __name__ == '__main__':
    unittest.main()
