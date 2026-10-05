"""Reconstruct Llama-style SentencePiece BPE from standalone GGUF metadata.

Score-ordered merge reconstruction follows the approach in Apache-2.0
Transformers convert_slow_tokenizer.generate_merges; runtime uses tokenizers
only. GGUF whitespace/special-token behavior is checked against llama.cpp's
MIT-licensed llm_tokenizer_spm_session, pin 836d57176dc699a726c55418e4f96b8ca628e1bf.
This supports the Llama SPM BPE convention, not arbitrary SentencePiece models.
"""
import math

from tokenizers import Tokenizer, decoders, models, normalizers


def build(metadata):
    if metadata.get('tokenizer.ggml.pre', 'default') not in ('default', 'llama'):
        raise ValueError('unsupported GGUF SentencePiece pre-tokenizer')
    if metadata.get('tokenizer.ggml.remove_extra_whitespaces', False) is not False:
        raise ValueError('GGUF SentencePiece whitespace normalization is unsupported')
    if (metadata.get('tokenizer.ggml.precompiled_charsmap')
            or any(key.startswith('tokenizer.ggml.normalizer.') and value is not False
                   for key, value in metadata.items())):
        raise ValueError('GGUF SentencePiece custom normalization is unsupported')
    prefix = metadata.get('tokenizer.ggml.add_space_prefix', True)
    if type(prefix) is not bool:
        raise ValueError('GGUF SentencePiece prefix flag must be boolean')
    tokens = metadata.get('tokenizer.ggml.tokens')
    types = metadata.get('tokenizer.ggml.token_type')
    scores = metadata.get('tokenizer.ggml.scores')
    if (not isinstance(tokens, list) or not tokens or not isinstance(types, list)
            or not isinstance(scores, list) or len(tokens) != len(types) or len(tokens) != len(scores)):
        raise ValueError('GGUF SentencePiece requires tokens, types and scores')
    if (any(not isinstance(t, str) or not t or len(t) > 4096 for t in tokens)
            or any(type(kind) is not int or kind not in (1, 2, 3, 4, 5, 6) for kind in types)
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
        raise ValueError('invalid GGUF SentencePiece vocabulary')
    vocab = {text: i for i, text in enumerate(tokens) if types[i] != 5}
    if len(vocab) != sum(kind != 5 for kind in types):
        raise ValueError('duplicate active GGUF SentencePiece tokens')
    unknown = metadata.get('tokenizer.ggml.unknown_token_id', 0)
    if (type(unknown) is not int or not 0 <= unknown < len(tokens) or types[unknown] not in (2, 3)
            or any(kind == 2 and i != unknown for i, kind in enumerate(types))):
        raise ValueError('invalid GGUF SentencePiece unknown token')
    if {text for text, kind in zip(tokens, types) if kind == 6} != {f'<0x{i:02X}>' for i in range(256)}:
        raise ValueError('GGUF SentencePiece requires all 256 byte fallback tokens')
    # Multiple splits of one piece share its score. Stable token-ID ordering,
    # then score and split lengths, preserves the established conversion order.
    merges = []
    scored_pieces = {}
    for text, score, kind in zip(tokens, scores, types):
        if kind != 1:
            continue
        splits = [(text[:i], text[i:]) for i in range(1, len(text))
                  if text[:i] in vocab and text[i:] in vocab]
        if splits:
            previous = scored_pieces.get(score)
            if previous is not None and not (set(previous) == {'▁'} and set(text) == {'▁'}):
                raise ValueError('GGUF SentencePiece tied merge scores require positional BPE support')
            scored_pieces[score] = text
        splits.sort(key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        merges.extend((left, right, score) for left, right in splits)
    merges.sort(key=lambda item: (item[2], len(item[0]), len(item[1])), reverse=True)
    result = Tokenizer(models.BPE(vocab, [(a, b) for a, b, _ in merges],
                                  unk_token=tokens[unknown], byte_fallback=True, fuse_unk=True))
    normalization = [normalizers.Prepend('▁')] if prefix else []
    normalization.append(normalizers.Replace(' ', '▁'))
    result.normalizer = normalizers.Sequence(normalization)
    decoding = [decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse()]
    if prefix:
        decoding.append(decoders.Strip(' ', 1, 0))
    result.decoder = decoders.Sequence(decoding)
    return result, tokens, types, vocab, True


class SentencePieceTokenizer:
    """Preserve GGUF's leading-space rule when an initial control token is hidden.

    Rust still performs vocabulary lookup and decoding. A decode-only view shares
    the model and removes just the Strip stage; normal generated-token sequences
    use the original tokenizer directly. The backend remains available to grammar
    libraries that require an actual tokenizers.Tokenizer instance.
    """
    def __init__(self, backend):
        self.backend_tokenizer = backend
        added = backend.get_added_tokens_decoder()
        self._special_ids = frozenset(i for i, token in added.items() if token.special)
        self._unstripped = Tokenizer(backend.model)
        self._unstripped.add_tokens(list(added.values()))
        self._unstripped.decoder = decoders.Sequence([
            decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse()])

    def __getattr__(self, name):
        return getattr(self.backend_tokenizer, name)

    def decode(self, ids, skip_special_tokens=True):
        decoder = (self._unstripped if skip_special_tokens and len(ids) and ids[0] in self._special_ids
                   else self.backend_tokenizer)
        return decoder.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]
