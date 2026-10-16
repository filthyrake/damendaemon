import json
from pathlib import Path
import unittest

from dllm.gguf_model import tokenizer
from dllm.gguf_sentencepiece import restore_tokenizer
from tests.reference.generate_gguf_bpe_specials import metadata


class BpeSpecialTests(unittest.TestCase):
    def test_independent_reference_and_serialized_batch(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/gguf-bpe-specials.json').read_text())
        for case in fixture['cases']:
            original, _, _ = tokenizer(metadata(case['pre'], case['kind'], case['eos']))
            for candidate in (original, restore_tokenizer(original.to_str())):
                for sample in case['samples']:
                    with self.subTest(pre=case['pre'], kind=case['kind'], eos=case['eos'], sample=sample):
                        ids = candidate.encode(sample['text'], add_special_tokens=sample['add_special']).ids
                        self.assertEqual(ids, sample['expected']['ids'])
                        self.assertEqual(candidate.encode_batch([sample['text']], add_special_tokens=sample['add_special'])[0].ids, ids)
                        for skip, suffix in ((True, 'skip'), (False, 'all')):
                            decoded = candidate.decode(ids, skip_special_tokens=skip).encode().hex()
                            self.assertEqual(decoded, sample['expected']['decode_' + suffix])
                            self.assertEqual(decoded, sample['expected']['generated_' + suffix])

    def test_literal_special_reference_full_batch_and_stream(self):
        from dllm.bytelevel_stream import ByteLevelStream
        from dllm.logprobs import TokenText
        from tests.reference.generate_gguf_bpe_literals import metadata as literal_metadata
        fixture = json.loads((Path(__file__).parent / 'fixtures/gguf-bpe-literals.json').read_text())
        for case in fixture['cases']:
            original, _, _ = tokenizer(literal_metadata(case['pre'], case['kind'], case['literal']))
            for candidate in (original, restore_tokenizer(original.to_str())):
                self.assertEqual(TokenText(candidate).token(257)[0], case['literal'])
                for sample in case['samples']:
                    with self.subTest(pre=case['pre'], kind=case['kind'], literal=case['literal'], sample=sample):
                        encoded = candidate.encode(sample['text'], add_special_tokens=False).ids
                        self.assertEqual(encoded, sample['expected']['ids'])
                        ids = encoded if sample['explicit'] is None else sample['explicit']
                        for skip, suffix in ((True, 'skip'), (False, 'all')):
                            # The C API emits raw bytes; public text decoding replaces incomplete UTF-8.
                            expected = bytes.fromhex(sample['expected']['decode_' + suffix]).decode(errors='replace')
                            self.assertEqual(candidate.decode(ids, skip_special_tokens=skip), expected)
                            self.assertEqual(candidate.decode_batch([ids], skip_special_tokens=skip), [expected])
                            stream = ByteLevelStream(candidate, ('<|endoftext|>', '<s>', 'bcx', case['literal']))
                            hidden = {i for i,t in candidate.get_added_tokens_decoder().items() if skip and t.special}
                            self.assertEqual(''.join(stream.feed(i) for i in ids if i not in hidden) + stream.finish(), expected)

    def test_unused_independent_reference_and_stream(self):
        from types import SimpleNamespace
        from dllm.bytelevel_stream import ByteLevelStream
        from dllm.logprobs import TokenText
        from dllm.structured import Constraint
        from tests.reference.generate_gguf_bpe_unused import metadata as unused_metadata
        fixture = json.loads((Path(__file__).parent / 'fixtures/gguf-bpe-unused.json').read_text())
        for case in fixture['cases']:
            source = unused_metadata(case['pre'], case['unused'], case['eos'])
            original, _, _ = tokenizer(source)
            for candidate in (original, restore_tokenizer(original.to_str())):
                for index in case['unused']:
                    self.assertEqual(TokenText(candidate).token(index), ('', ()))
                with self.assertRaisesRegex(ValueError, 'suppressed GGUF'):
                    Constraint(SimpleNamespace(tokenizer=candidate), '{}')
                for sample in case['samples']:
                    with self.subTest(pre=case['pre'], unused=case['unused'], eos=case['eos'], sample=sample):
                        ids = candidate.encode(sample['text'], add_special_tokens=sample['add_special']).ids
                        self.assertEqual(ids, sample['expected']['ids'])
                        ids = ids if sample['explicit'] is None else sample['explicit']
                        for skip, suffix in ((True, 'skip'), (False, 'all')):
                            decoded = candidate.decode(ids, skip_special_tokens=skip).encode().hex()
                            self.assertEqual(decoded, sample['expected']['decode_' + suffix])
                            self.assertEqual(decoded, sample['expected']['generated_' + suffix])
                            self.assertEqual(candidate.decode_batch([ids], skip_special_tokens=skip), [bytes.fromhex(decoded).decode()])
                        stream = ByteLevelStream(candidate, ('<user>',))
                        # Control pieces are omitted by serving, not fed into the decoder.
                        visible = [i for i in ids if i not in (256, 262)]
                        incremental = ''.join(stream.feed(i) for i in visible) + stream.finish()
                        self.assertEqual(incremental.encode().hex(), sample['expected']['decode_skip'])

    def test_unused_duplicate_and_eos_ids_rejected(self):
        from tests.reference.generate_gguf_bpe_unused import metadata as unused_metadata
        source = unused_metadata('qwen2', [258], False)
        source['tokenizer.ggml.tokens'][258] = 'aa'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            tokenizer(source)
        source = unused_metadata('qwen2', [256], False)
        with self.assertRaisesRegex(ValueError, 'eos token ID'):
            tokenizer(source)

    def test_ambiguous_equal_length_specials_rejected(self):
        for pre in ('qwen2', 'llama-bpe'):
            source = metadata(pre, 4, False)
            source['tokenizer.ggml.tokens'][258] = 'bc'
            with self.assertRaisesRegex(ValueError, 'equal-length overlapping special'):
                tokenizer(source)

    def test_bytelevel_stream_and_logprob_presentation(self):
        from dllm.bytelevel_stream import ByteLevelStream
        from dllm.logprobs import TokenText
        original, _, _ = tokenizer(metadata('qwen2', 4, False))
        for candidate in (original, restore_tokenizer(original.to_str())):
            stream = ByteLevelStream(candidate, ('ab', 'bcx'))
            text = 'café 😀 abcx'
            ids = candidate.encode(text, add_special_tokens=False).ids
            self.assertEqual(''.join(stream.feed(i) for i in ids) + stream.finish(), text)
            self.assertEqual(TokenText(candidate).token(258), ('bcx', (98, 99, 120)))


if __name__ == '__main__':
    unittest.main()
