# Score-ordered merge reconstruction adapted from Transformers 4.51.3.
# Copyright 2018 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0; see
# docs/licenses/transformers-Apache-2.0.txt. Modifications: read GGUF metadata,
# validate the supported vocabulary subset, preserve GGUF decoding semantics.
"""Reconstruct Llama-style SentencePiece BPE from standalone GGUF metadata.

Score-ordered merge reconstruction follows the approach in Apache-2.0
Transformers convert_slow_tokenizer.generate_merges. Runtime uses Rust tokenizers
for unique scores and native C++ SentencePiece for positional score ties. GGUF whitespace/special-token behavior is checked against llama.cpp's
MIT-licensed llm_tokenizer_spm_session, pin 836d57176dc699a726c55418e4f96b8ca628e1bf.
This supports the Llama SPM BPE convention, not arbitrary SentencePiece models.
"""
import base64
import json
import math
from types import SimpleNamespace

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
    positional = False
    for text, score, kind in zip(tokens, scores, types):
        if kind != 1:
            continue
        splits = [(text[:i], text[i:]) for i in range(1, len(text))
                  if text[:i] in vocab and text[i:] in vocab]
        if splits:
            previous = scored_pieces.get(score)
            if previous is not None and not (set(previous) == {'▁'} and set(text) == {'▁'}):
                positional = True
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
    native = native_model(tokens, types, scores, unknown) if positional else None
    return result, tokens, types, vocab, True, native


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

    def decode_generated(self, ids, skip_special_tokens=True):
        # A continuation has no artificial beginning-of-text prefix to remove.
        # Match llama.cpp server token-to-piece output, not full detokenization.
        return self._unstripped.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]


def native_model(tokens, types, scores, unknown):
    """Build an identity-normalized BPE model for native positional score ties."""
    try:
        from sentencepiece import sentencepiece_model_pb2 as proto
    except ImportError as error:
        raise ValueError('GGUF tied SentencePiece scores require dllm[gguf] native tokenizer dependencies') from error
    if 5 in types:
        raise ValueError("positional SentencePiece with unused tokens is not yet supported")
    model = proto.ModelProto()
    model.trainer_spec.model_type = proto.TrainerSpec.BPE
    model.trainer_spec.vocab_size = len(tokens)
    model.trainer_spec.byte_fallback = True
    model.trainer_spec.unk_id = unknown
    model.trainer_spec.unk_piece = tokens[unknown]
    # These tokens are added explicitly using the GGUF postprocessor policy.
    model.trainer_spec.bos_id = -1
    model.trainer_spec.eos_id = -1
    model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = 'identity'
    model.normalizer_spec.add_dummy_prefix = False
    model.normalizer_spec.remove_extra_whitespaces = False
    model.normalizer_spec.escape_whitespaces = False
    for index, (text, kind, score) in enumerate(zip(tokens, types, scores)):
        piece = model.pieces.add()
        piece.piece, piece.score = text, score
        piece.type = proto.ModelProto.SentencePiece.UNKNOWN if index == unknown else kind
    return model.SerializeToString()


class PositionalSentencePieceTokenizer(SentencePieceTokenizer):
    """Native C++ score BPE, with explicit GGUF special-token boundaries.

    Rust retains ID lookup and byte decoding. Its static-rank encoder is never
    used for prompts on this path; serialized state preserves the native model.
    """
    def __init__(self, backend, model, prefix=True):
        super().__init__(backend)
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError as error:
            raise ValueError('GGUF tied SentencePiece scores require dllm[gguf] native tokenizer dependencies') from error
        self.native_model = model
        self.space_prefix = prefix
        self._processor = SentencePieceProcessor(model_proto=model)
        self._pieces = {token.content: index for index, token in backend.get_added_tokens_decoder().items()}
        self._special_order = sorted(self._pieces, key=lambda text: len(text.encode('utf-8')), reverse=True)
        policy = json.loads(backend.to_str()).get('post_processor')
        self._prefix, self._suffix = [], []
        if policy:
            target = self._prefix
            for entry in policy['single']:
                if 'Sequence' in entry:
                    target = self._suffix
                else:
                    target.extend(policy['special_tokens'][entry['SpecialToken']['id']]['ids'])

    def encode(self, text, add_special_tokens=True):
        if not isinstance(text, str):
            raise TypeError('GGUF tokenizer input must be text')
        ids = list(self._prefix) if add_special_tokens else []
        # Match GGUF's global longest-special-first partitioning, including a
        # longer special beginning inside a shorter one. Leftmost regex matching
        # would incorrectly consume the shorter special in that case.
        fragments = [text]
        for piece in self._special_order:
            divided = []
            for fragment in fragments:
                if isinstance(fragment, int):
                    divided.append(fragment)
                    continue
                parts = fragment.split(piece)
                for index, part in enumerate(parts):
                    if index:
                        divided.append(self._pieces[piece])
                    if part:
                        divided.append(part)
            fragments = divided
        for fragment in fragments:
            if isinstance(fragment, int):
                ids.append(fragment)
                continue
            if not fragment:
                continue
            normalized = ('▁' if self.space_prefix else '') + fragment.replace(' ', '▁')
            ids.extend(self._processor.encode(normalized, out_type=int))
        if add_special_tokens:
            ids.extend(self._suffix)
        return SimpleNamespace(ids=ids)

    def encode_batch(self, sequences, add_special_tokens=True):
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in sequences]

    def to_str(self, pretty=False):
        return json.dumps(dict(dllm_sentencepiece=1, backend=self.backend_tokenizer.to_str(),
            model=base64.b64encode(self.native_model).decode('ascii'), prefix=self.space_prefix),
            indent=2 if pretty else None)


def restore_tokenizer(serialized, prefix=False):
    """Restore both static-rank and native positional GGUF tokenizer state."""
    state = json.loads(serialized)
    if state.get('dllm_sentencepiece') == 1:
        return PositionalSentencePieceTokenizer(Tokenizer.from_str(state['backend']),
            base64.b64decode(state['model'], validate=True), state['prefix'])
    backend = Tokenizer.from_str(serialized)
    return SentencePieceTokenizer(backend) if prefix else backend
