"""Standalone Qwen2/Qwen3 GGUF import into dllm's own decoder and kernels."""
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors

from .config import decoder_config
from .device import selected_device
from .gguf_reader import BoundedReader
from .qwen2 import ModelWeights
from .quantization import dequantize, dequantize_q4, mode


# Qwen2 BPE pre-tokenization used by GGUF's qwen2 tokenizer. GGUF does not
# specify the NFC normalizer present in some HF tokenizer.json snapshots.
QWEN_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple
    kind: int
    offset: int
    length: int

    @property
    def host_bytes(self):
        return self.length if self.kind in (2, 8) else math.prod(self.shape) * 2


@dataclass(frozen=True)
class Plan:
    path: Path
    identity: tuple
    config: dict
    tensors: dict
    tokenizer_json: str
    template: str | None
    special_tokens: dict
    eos: frozenset
    quantization: str
    host_bytes: int


def tokenizer(metadata):
    if metadata.get('tokenizer.ggml.model') != 'gpt2' or metadata.get('tokenizer.ggml.pre') != 'qwen2':
        raise ValueError('GGUF requires the qwen2 GPT-2 BPE tokenizer')
    tokens = metadata.get('tokenizer.ggml.tokens')
    types = metadata.get('tokenizer.ggml.token_type')
    merges = metadata.get('tokenizer.ggml.merges')
    if (not isinstance(tokens, list) or not tokens or not isinstance(types, list)
            or len(types) != len(tokens) or not isinstance(merges, list)):
        raise ValueError('GGUF tokenizer requires tokens, token types and BPE merges')
    if any(not isinstance(t, str) or not t for t in tokens) or any(t not in (1, 3, 4, 5) for t in types):
        raise ValueError('unsupported GGUF token or token type')
    vocab = {t: i for i, t in enumerate(tokens) if types[i] != 5}
    if len(vocab) != sum(t != 5 for t in types):
        raise ValueError('duplicate active GGUF token strings')
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(vocab):
        raise ValueError('GGUF byte-level tokenizer is missing byte tokens')
    if metadata.get('tokenizer.ggml.add_space_prefix', False):
        raise ValueError('GGUF qwen2 tokenizer cannot add a space prefix')
    pairs = [m.split(' ') for m in merges if isinstance(m, str)]
    if len(pairs) != len(merges) or any(len(pair) != 2 for pair in pairs):
        raise ValueError('invalid GGUF BPE merge pair')
    if any(pair[0] not in vocab or pair[1] not in vocab or ''.join(pair) not in vocab for pair in pairs):
        raise ValueError('GGUF BPE merge references an unknown token')
    result = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(p) for p in pairs]))
    result.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    result.decoder = decoders.ByteLevel()
    for i, (text, kind) in enumerate(zip(tokens, types)):
        if kind in (3, 4):
            result.add_tokens([AddedToken(text, normalized=False, special=kind == 3)])
            if result.token_to_id(text) != i:
                raise ValueError('GGUF added token changed its declared token ID')
    special = {}
    eos = set()
    for name in ('bos', 'eos', 'eot', 'eom'):
        index = metadata.get(f'tokenizer.ggml.{name}_token_id')
        if index is not None:
            if type(index) is not int or not 0 <= index < len(tokens) or types[index] == 5:
                raise ValueError(f'invalid GGUF {name} token ID')
            if name in ('bos', 'eos'):
                special[f'{name}_token'] = tokens[index]
            if name != 'bos':
                eos.add(index)
    # Qwen marks both end-of-text and end-of-turn as generation stops.
    for text in ('<|im_end|>', '<|endoftext|>'):
        if text in vocab:
            eos.add(vocab[text])
    if not eos:
        raise ValueError('GGUF model needs an EOS token')
    prefix, suffix, added = [], [], []
    for name, target in (('bos', prefix), ('eos', suffix)):
        enabled = metadata.get(f'tokenizer.ggml.add_{name}_token', False)
        if type(enabled) is not bool:
            raise ValueError('GGUF automatic special-token flags must be boolean')
        if enabled:
            text = special.get(f'{name}_token')
            if text is None:
                raise ValueError(f'GGUF adds {name} but supplies no token ID')
            target.append(text)
            added.append((text, vocab[text]))
    if added:
        result.post_processor = processors.TemplateProcessing(single=prefix + ['$A'] + suffix, special_tokens=added)
    return result, special, frozenset(eos)


def model_config(metadata, tensors, vocab_size, eos):
    arch = metadata.get('general.architecture')
    if arch not in ('qwen2', 'qwen3'):
        raise ValueError('GGUF currently supports dense qwen2 and qwen3 only')
    if any(name.startswith('split.') for name in metadata):
        raise ValueError('split GGUF files are not supported')
    def get(name, default=None):
        return metadata.get(f'{arch}.{name}', default)
    if (get('rope.scaling.type', 'none') != 'none'
            or any(name.startswith(f'{arch}.rope.scaling.') and name != f'{arch}.rope.scaling.type' for name in metadata)
            or get('attention.sliding_window', 0) != 0
            or get('expert_count', 0) != 0):
        raise ValueError('GGUF scaled RoPE, windowed attention and experts are unsupported')
    c = dict(model_type=arch, hidden_act='silu', attention_bias=arch == 'qwen2',
             hidden_size=get('embedding_length'), intermediate_size=get('feed_forward_length'),
             num_hidden_layers=get('block_count'), max_position_embeddings=get('context_length'),
             num_attention_heads=get('attention.head_count'), num_key_value_heads=get('attention.head_count_kv'),
             rope_theta=get('rope.freq_base', 10000.0), rms_norm_eps=get('attention.layer_norm_rms_epsilon'),
             vocab_size=vocab_size, eos_token_id=sorted(eos), tie_word_embeddings='output.weight' not in tensors)
    if get('attention.key_length') is not None:
        c['head_dim'] = get('attention.key_length')
    c = decoder_config(c)
    if get('attention.value_length', c['head_dim']) != c['head_dim'] or get('rope.dimension_count', c['head_dim']) != c['head_dim']:
        raise ValueError('GGUF requires matching full-head Q/K/V and rotary dimensions')
    return c


def tensor_groups(c, tensors):
    """Return destination names, source names and expected *logical* shapes."""
    h, n, d, kv, f = (c[k] for k in ('hidden_size', 'num_attention_heads', 'head_dim', 'num_key_value_heads', 'intermediate_size'))
    yield 'embed', [('token_embd.weight', (c['vocab_size'], h))]
    yield 'norm', [('output_norm.weight', (h,))]
    if 'output.weight' in tensors:
        yield 'head', [('output.weight', (c['vocab_size'], h))]
    for i in range(c['num_hidden_layers']):
        p = f'blk.{i}.'
        yield f'{i}.qkv.weight', [(p + f'attn_{a}.weight', (rows * d, h)) for a, rows in (('q', n), ('k', kv), ('v', kv))]
        if c['model_type'] == 'qwen2':
            yield f'{i}.qkv.bias', [(p + f'attn_{a}.bias', (rows * d,)) for a, rows in (('q', n), ('k', kv), ('v', kv))]
        else:
            yield f'{i}.q_norm', [(p + 'attn_q_norm.weight', (d,))]
            yield f'{i}.k_norm', [(p + 'attn_k_norm.weight', (d,))]
        yield f'{i}.out', [(p + 'attn_output.weight', (h, n * d))]
        yield f'{i}.gate_up', [(p + f'ffn_{a}.weight', (f, h)) for a in ('gate', 'up')]
        yield f'{i}.down', [(p + 'ffn_down.weight', (h, f))]
        yield f'{i}.norm1', [(p + 'attn_norm.weight', (h,))]
        yield f'{i}.norm2', [(p + 'ffn_norm.weight', (h,))]


def plan(path):
    path = Path(path).resolve()
    return _plan(path, identity(path))


@lru_cache(maxsize=2)
def _plan(path, version):
    reader = BoundedReader(path)
    metadata = {name: field.contents() for name, field in reader.fields.items()}
    tensors = {t.name: TensorInfo(t.name, tuple(int(n) for n in reversed(t.shape)), int(t.tensor_type), int(t.data_offset), int(t.n_bytes)) for t in reader.tensors}
    # The cache retains metadata, never memory maps or weight arrays.
    del reader
    tok, special, eos = tokenizer(metadata)
    c = model_config(metadata, tensors, len(metadata['tokenizer.ggml.tokens']), eos)
    expected = set()
    for destination, group in tensor_groups(c, tensors):
        kinds = set()
        for name, shape in group:
            tensor = tensors.get(name)
            if tensor is None or tensor.shape != shape:
                raise ValueError(f'GGUF tensor {name} is missing or has the wrong shape')
            expected.add(name)
            kinds.add(tensor.kind if tensor.kind in (2, 8) else 0)
        if len(kinds) > 1:
            raise ValueError(f'GGUF fused projection {destination} has incompatible weight formats')
    if set(tensors) != expected:
        raise ValueError(f'unrecognized GGUF tensors: {sorted(set(tensors) - expected)[:4]}')
    kinds = {t.kind for t in tensors.values()} & {2, 8}
    quantization = 'mixed' if len(kinds) > 1 else 'q4_0' if 2 in kinds else 'q8_0' if 8 in kinds else 'none'
    total = sum(t.host_bytes for t in tensors.values())
    embedding = tensors['token_embd.weight']
    if embedding.kind in (2, 8):
        # Lookup requires dense FP16 embeddings; an absent head shares the
        # original packed matrix, not the expanded embedding.
        total += math.prod(embedding.shape) * 2
        if not c['tie_word_embeddings']:
            total -= embedding.host_bytes
    template = metadata.get('tokenizer.chat_template')
    if template is not None and not isinstance(template, str):
        raise ValueError('GGUF needs a single text chat template')
    if identity(path) != version:
        raise ValueError('GGUF file changed while reading metadata')
    return Plan(path, version, c, tensors, tok.to_str(), template, special, eos, quantization, total)


def weight_bytes(path, quantization=None):
    if mode(quantization) != 'none':
        raise ValueError('GGUF preserves checkpoint quantization; requested conversion is unsupported')
    return plan(path).host_bytes


def read_tensor(file, tensor):
    file.seek(tensor.offset)
    raw = np.fromfile(file, dtype=np.uint8, count=tensor.length)
    if raw.size != tensor.length:
        raise ValueError(f'truncated GGUF tensor: {tensor.name}')
    if tensor.kind in (0, 1):
        dtype = '<f4' if tensor.kind == 0 else '<f2'
        value = torch.from_numpy(raw.view(dtype).reshape(tensor.shape)).to(torch.float16)
        if not torch.isfinite(value).all():
            raise ValueError(f'GGUF tensor exceeds finite FP16 range: {tensor.name}')
        return value, None
    block_bytes = 18 if tensor.kind == 2 else 34
    blocks = raw.reshape(-1, block_bytes)
    scales = torch.from_numpy(blocks[:, :2].copy().view('<f2').reshape(tensor.shape[0], -1))
    if not torch.isfinite(scales).all():
        raise ValueError(f'GGUF tensor has non-finite scales: {tensor.name}')
    packed = blocks[:, 2:].copy()
    packed = packed if tensor.kind == 2 else packed.view(np.int8)
    return torch.from_numpy(packed.reshape(tensor.shape[0], -1)), scales


class GgufWeights(ModelWeights):
    def __init__(self, path, max_context, quantization=None):
        weight_bytes(path, quantization)
        source = plan(path)
        self.path, self.config = source.path, dict(source.config)
        self.quantization = source.quantization
        self.weight_format = 'gguf_' + ('f16' if self.quantization == 'none' else self.quantization)
        self.weight_cache_status = 'checkpoint-native'
        if selected_device() == 'mps' and any(t.kind == 2 for t in source.tensors.values()):
            raise ValueError('GGUF Q4_0 Metal kernels are not implemented')
        if not 1 <= max_context <= self.config['max_position_embeddings']:
            raise ValueError('context exceeds model position limit')
        self.max_context = max_context
        self.configure_tokenizer(Tokenizer.from_str(source.tokenizer_json), source.template, source.special_tokens)
        self.eos = set(source.eos)
        weights, scales = {}, {}
        with self.path.open('rb') as file:
            for destination, group in tensor_groups(self.config, source.tensors):
                parts = [read_tensor(file, source.tensors[name]) for name, _ in group]
                values = [p[0] for p in parts]
                weights[destination] = values[0] if len(values) == 1 else torch.cat(values)
                if parts[0][1] is not None:
                    parts_scales = [p[1] for p in parts]
                    scales[destination] = parts_scales[0] if len(parts_scales) == 1 else torch.cat(parts_scales)
        if self.config['tie_word_embeddings']:
            weights['head'] = weights['embed']
            if 'embed' in scales:
                scales['head'] = scales['embed']
        if 'embed' in scales:
            unpack = dequantize_q4 if weights['embed'].dtype == torch.uint8 else dequantize
            # Bound temporary expansion by rows rather than making a FP32 copy
            # of the entire vocabulary matrix.
            packed, step = weights['embed'], scales.pop('embed')
            embedding = torch.empty(source.tensors['token_embd.weight'].shape, dtype=torch.float16)
            for first in range(0, len(embedding), 128):
                embedding[first:first+128] = unpack(packed[first:first+128], step[first:first+128], torch.float16)
            if not torch.isfinite(embedding).all():
                raise ValueError('GGUF embedding exceeds finite FP16 range')
            weights['embed'] = embedding
        if identity(self.path) != source.identity:
            raise ValueError('GGUF file changed while loading weights')
        self.finalize_weights(weights, scales)
        if self.bytes != source.host_bytes:
            raise ValueError('GGUF loaded storage disagrees with preflight estimate')
