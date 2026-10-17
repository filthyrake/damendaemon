"""Actual optional xgrammar CPU contracts; no model weights or GPU execution."""
import importlib.util
import json
from types import SimpleNamespace
import unittest

from damendaemon.gguf_model import tokenizer
from damendaemon.gguf_sentencepiece import restore_tokenizer
from damendaemon.structured import Constraint
from tests.reference.generate_gguf_bpe_literals import metadata as bpe_metadata
from tests.test_gguf_sentencepiece import metadata as spm_metadata


@unittest.skipUnless(importlib.util.find_spec('xgrammar'), 'optional xgrammar is not installed')
class GgufGrammarTests(unittest.TestCase):
    def cases(self):
        for pre in ('qwen2', 'llama-bpe'):
            source = bpe_metadata(pre, 4, '<é>')
            yield source, 257, '<é>', [source['tokenizer.ggml.tokens'].index(c) for c in ('Ã', '©')]
        source = spm_metadata()
        source['tokenizer.ggml.tokens'][265] = '▁user'
        yield source, 265, '▁user', [3 + 0xC3, 3 + 0xA9]

    def test_unused_and_positional_vocabularies(self):
        from tests.reference.generate_gguf_bpe_unused import metadata as unused_metadata
        sources = [unused_metadata(pre, [257], False) for pre in ('qwen2', 'llama-bpe')]
        for tied in (False, True):
            for suppressed in (False, True):
                for prefix in (False, True):
                    source = spm_metadata()
                    source['tokenizer.ggml.add_space_prefix'] = prefix
                    if tied:
                        source['tokenizer.ggml.scores'][264] = -5.
                    if suppressed:
                        source['tokenizer.ggml.token_type'][260] = 5
                    sources.append(source)
        for source in sources:
            original, _, eos = tokenizer(source)
            for candidate in (original, restore_tokenizer(original.to_str())):
                with self.subTest(native=hasattr(candidate, 'native_model'),
                                  suppressed=candidate.suppressed_token_ids,
                                  prefix=source.get('tokenizer.ggml.add_space_prefix')):
                    model = SimpleNamespace(tokenizer=candidate, eos=eos,
                        config={'vocab_size':len(source['tokenizer.ggml.tokens'])})
                    spm = source['tokenizer.ggml.model'] == 'llama'
                    pieces = (['<0x22>', 'ab', '<0xC3>', '<0xA9>', '▁', '<user>', '<0x22>']
                              if spm else ['"', 'ab', 'Ã', '©', 'Ġ', '<user>', '"'])
                    ids = [candidate.token_to_id(piece) for piece in pieces]
                    constraint = Constraint(model, json.dumps({'type':'string','const':'abé <user>'}))
                    hidden = set(candidate.suppressed_token_ids) | {
                        i for i, added in candidate.get_added_tokens_decoder().items()
                        if added.special and i not in eos}
                    import torch
                    logits = torch.zeros(model.config['vocab_size'])
                    for token in ids:
                        masked = constraint.apply(logits)
                        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
                        self.assertTrue(torch.isfinite(masked[token]))
                        for invisible in hidden:
                            self.assertTrue(torch.isneginf(masked[invisible]))
                        constraint.matcher.fill_next_token_bitmask(constraint.mask)
                        for invisible in hidden:
                            self.assertFalse(int(constraint.mask[0, invisible // 32]) & (1 << (invisible % 32)))
                        self.assertTrue(int(constraint.mask[0, token // 32]) & (1 << (token % 32)))
                        constraint.accept(token)
                    constraint.accept(next(iter(eos)))
                    self.assertTrue(constraint.matcher.is_terminated())
                    decode = getattr(candidate, 'decode_generated', candidate.decode)
                    self.assertEqual(decode(ids), '"abé <user>"')

    def test_ordinary_backend_path_still_accepts_json(self):
        source = bpe_metadata('qwen2', 4, '<user>')
        candidate, _, eos = tokenizer(source)
        self.assertFalse(candidate._literal_tokens)
        model = SimpleNamespace(tokenizer=candidate.backend_tokenizer, eos=eos,
            config={'vocab_size':len(source['tokenizer.ggml.tokens'])})
        constraint = Constraint(model, json.dumps({'type':'string','const':'é'}))
        for token in candidate.encode('"é"', add_special_tokens=False).ids:
            constraint.accept(token)
        constraint.accept(next(iter(eos)))
        self.assertTrue(constraint.matcher.is_terminated())

    def test_literal_and_partial_utf8_match_emitted_text(self):
        for source, literal_id, literal, utf8_ids in self.cases():
            original, _, eos = tokenizer(source)
            for candidate in (original, restore_tokenizer(original.to_str())):
                with self.subTest(family=source['tokenizer.ggml.model'], literal=literal):
                    model = SimpleNamespace(tokenizer=candidate, eos=eos,
                        config={'vocab_size':len(source['tokenizer.ggml.tokens']) + 2})
                    quote = candidate.token_to_id('"')
                    if quote is None:
                        quote = candidate.token_to_id('<0x22>')
                    constraint = Constraint(model, json.dumps({'type':'string','const':literal + 'é'}))
                    constraint.accept(quote)
                    # The real CPU mask must allow the same pieces the matcher accepts.
                    constraint.matcher.fill_next_token_bitmask(constraint.mask)
                    self.assertTrue(int(constraint.mask[0, literal_id // 32]) & (1 << (literal_id % 32)))
                    for token in [literal_id, *utf8_ids, quote]:
                        constraint.accept(token)
                    constraint.accept(next(iter(eos)))
                    self.assertTrue(constraint.matcher.is_terminated())
                    decode = getattr(candidate, 'decode_generated', candidate.decode)
                    self.assertEqual(decode([quote, literal_id, *utf8_ids, quote]), '"' + literal + 'é"')
                    wrong = Constraint(model, json.dumps({'type':'string','const':'different'}))
                    wrong.accept(quote)
                    self.assertFalse(wrong.matcher.accept_token(literal_id))
                    # Hidden non-EOS controls and padded model rows cannot satisfy grammar.
                    fresh = Constraint(model, json.dumps({'type':'string'}))
                    for token_id in (source['tokenizer.ggml.bos_token_id'],
                                     len(source['tokenizer.ggml.tokens'])):
                        self.assertFalse(fresh.matcher.accept_token(token_id))


if __name__ == '__main__':
    unittest.main()
