"""Standalone Qwen2/Qwen3/Llama GGUF import into damendaemon's own decoder and kernels."""
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import re
import stat
from pathlib import Path

import numpy as np
import torch
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors

from .config import decoder_config
from .device import selected_device
from .gguf_reader import BoundedReader,MAX_HEADER
from .gguf_packed import FORMATS, KQUANTS, NAMES, read_raw, layout as packed_layout
from .qwen2 import ModelWeights
from .quantization import dequantize, dequantize_q4, mode


# Qwen2 BPE pre-tokenization used by GGUF's qwen2 tokenizer. GGUF does not
# specify the NFC normalizer present in some HF tokenizer.json snapshots.
QWEN_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def identity(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('GGUF requires regular files')
    return file_identity(info)


def file_identity(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple
    kind: int
    offset: int
    length: int
    file_index: int = 0

    @property
    def host_bytes(self):
        return self.length if self.kind in FORMATS else math.prod(self.shape) * (4 if self.name == 'rope_freqs.weight' else 2)


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
    raw_groups: frozenset
    sentencepiece_prefix: bool = False


def bpe_tokenizer(metadata):
    pre = metadata.get('tokenizer.ggml.pre')
    llama = pre in ('llama-bpe', 'llama3', 'llama-v3')
    if metadata.get('tokenizer.ggml.model') != 'gpt2' or not (pre == 'qwen2' or llama):
        raise ValueError('GGUF requires a supported Qwen2 or Llama3 GPT-2 BPE tokenizer')
    tokens = metadata.get('tokenizer.ggml.tokens')
    types = metadata.get('tokenizer.ggml.token_type')
    merges = metadata.get('tokenizer.ggml.merges')
    if (not isinstance(tokens, list) or not tokens or not isinstance(types, list)
            or len(types) != len(tokens) or not isinstance(merges, list)):
        raise ValueError('GGUF tokenizer requires tokens, token types and BPE merges')
    if any(not isinstance(t, str) or not t for t in tokens) or any(t not in (1, 3, 4, 5) for t in types):
        raise ValueError('unsupported GGUF token or token type')
    vocab = {t: i for i, t in enumerate(tokens)}
    if len(vocab) != len(tokens):
        raise ValueError('duplicate GGUF token strings')
    from .gguf_tokenizer import validate_special_overlaps
    validate_special_overlaps(tokens, types)
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(vocab):
        raise ValueError('GGUF byte-level tokenizer is missing byte tokens')
    if metadata.get('tokenizer.ggml.add_space_prefix', False):
        raise ValueError('GGUF tokenizer cannot add a space prefix')
    pairs = [m.split(' ') for m in merges if isinstance(m, str)]
    if len(pairs) != len(merges) or any(len(pair) != 2 for pair in pairs):
        raise ValueError('invalid GGUF BPE merge pair')
    if any(pair[0] not in vocab or pair[1] not in vocab or ''.join(pair) not in vocab for pair in pairs):
        raise ValueError('GGUF BPE merge references an unknown token')
    result = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(p) for p in pairs], ignore_merges=llama))
    pattern = QWEN_PATTERN.replace(r'|\p{N}|', r'|\p{N}{1,3}|') if llama else QWEN_PATTERN
    result.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    result.decoder = decoders.ByteLevel()
    return result, tokens, types, vocab, llama


def tokenizer(metadata):
    sentencepiece = metadata.get('tokenizer.ggml.model') == 'llama'
    native = None
    if sentencepiece:
        if metadata.get('general.architecture') != 'llama':
            raise ValueError('GGUF SentencePiece currently requires the llama architecture')
        from .gguf_sentencepiece import build
        result, tokens, types, vocab, llama, native = build(metadata)
    else:
        result, tokens, types, vocab, llama = bpe_tokenizer(metadata)
    for i, (text, kind) in enumerate(zip(tokens, types)):
        if kind in (3, 4) or (sentencepiece and kind == 2):
            result.add_tokens([AddedToken(text, normalized=False, special=kind != 4)])
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
    # Generation stops include end-of-text, end-of-turn and end-of-message.
    for text in ('<|im_end|>', '<|endoftext|>', '<|end_of_text|>', '<|eot_id|>', '<|eom_id|>'):
        if text in vocab and types[vocab[text]] != 5:
            eos.add(vocab[text])
    if not eos:
        raise ValueError('GGUF model needs an EOS token')
    prefix, suffix, added = [], [], []
    for name, target in (('bos', prefix), ('eos', suffix)):
        enabled = metadata.get(f'tokenizer.ggml.add_{name}_token', llama and name == 'bos')
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
    suppressed = {i for i, kind in enumerate(types) if kind == 5}
    if native is not None:
        from .gguf_sentencepiece import PositionalSentencePieceTokenizer
        result = PositionalSentencePieceTokenizer(result, native, metadata.get('tokenizer.ggml.add_space_prefix', True), suppressed)
    elif sentencepiece:
        from .gguf_sentencepiece import SentencePieceTokenizer
        result = SentencePieceTokenizer(result, suppressed)
    else:
        from .gguf_tokenizer import SpecialTokenTokenizer
        result = SpecialTokenTokenizer(result, suppressed)
    return result, special, frozenset(eos)


def model_config(metadata, tensors, vocab_size, eos):
    arch = metadata.get('general.architecture')
    if arch not in ('qwen2', 'qwen3', 'llama'):
        raise ValueError('GGUF currently supports dense qwen2, qwen3 and llama only')
    if any(name.startswith('split.') for name in metadata):
        raise ValueError('split GGUF files are not supported')
    def get(name, default=None):
        return metadata.get(f'{arch}.{name}', default)
    if get('attention.causal', True) is not True:
        raise ValueError('GGUF decoder requires causal attention')
    window = get('attention.sliding_window', 0)
    if type(window) is not int or window < 0:
        raise ValueError('GGUF sliding window must be a nonnegative integer')
    if (window and arch != 'llama') or get('attention.sliding_window_pattern') is not None:
        raise ValueError('GGUF only supports uniform Llama/Mistral sliding-window attention')
    scaling = {name.removeprefix(f'{arch}.rope.scaling.'): value
               for name, value in metadata.items() if name.startswith(f'{arch}.rope.scaling.')}
    if f'{arch}.rope.scale_linear' in metadata:
        if scaling:
            raise ValueError('GGUF mixes legacy and modern RoPE scaling metadata')
        scaling = {'type': 'linear', 'factor': get('rope.scale_linear')}
    kind = scaling.get('type', 'none')
    if kind not in ('none', 'linear') or set(scaling) - ({'type', 'factor'} if kind == 'linear' else {'type'}):
        raise ValueError('GGUF only supports explicit linear RoPE scaling')
    if get('expert_count', 0) != 0:
        raise ValueError('GGUF experts are unsupported')
    # Uniform windows use the existing Mistral execution policy. The GGUF
    # tensor layout still uses Llama's interleaved rotary rows on disk.
    c = dict(model_type='mistral' if window else arch, hidden_act='silu', attention_bias=arch == 'qwen2',
             hidden_size=get('embedding_length'), intermediate_size=get('feed_forward_length'),
             num_hidden_layers=get('block_count'), max_position_embeddings=get('context_length'),
             num_attention_heads=get('attention.head_count'), num_key_value_heads=get('attention.head_count_kv'),
             rope_theta=get('rope.freq_base', 10000.0), rms_norm_eps=get('attention.layer_norm_rms_epsilon'),
             vocab_size=vocab_size, eos_token_id=sorted(eos), tie_word_embeddings='output.weight' not in tensors)
    if kind == 'linear':
        c['rope_scaling'] = {'rope_type': 'linear', 'factor': scaling.get('factor')}
    if window:
        c['sliding_window'] = window
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
    if c['model_type'] == 'llama' and 'rope_freqs.weight' in tensors:
        yield 'rope_factors', [('rope_freqs.weight', (d // 2,))]
    if 'output.weight' in tensors:
        yield 'head', [('output.weight', (c['vocab_size'], h))]
    for i in range(c['num_hidden_layers']):
        p = f'blk.{i}.'
        yield f'{i}.qkv.weight', [(p + f'attn_{a}.weight', (rows * d, h)) for a, rows in (('q', n), ('k', kv), ('v', kv))]
        if c['model_type'] == 'qwen2':
            yield f'{i}.qkv.bias', [(p + f'attn_{a}.bias', (rows * d,)) for a, rows in (('q', n), ('k', kv), ('v', kv))]
        elif c['model_type'] == 'qwen3':
            yield f'{i}.q_norm', [(p + 'attn_q_norm.weight', (d,))]
            yield f'{i}.k_norm', [(p + 'attn_k_norm.weight', (d,))]
        yield f'{i}.out', [(p + 'attn_output.weight', (h, n * d))]
        yield f'{i}.gate_up', [(p + f'ffn_{a}.weight', (f, h)) for a in ('gate', 'up')]
        yield f'{i}.down', [(p + 'ffn_down.weight', (h, f))]
        yield f'{i}.norm1', [(p + 'attn_norm.weight', (h,))]
        yield f'{i}.norm2', [(p + 'ffn_norm.weight', (h,))]


SPLIT_KEYS=frozenset(('split.no','split.count','split.tensors.count'))


def split_metadata(metadata,index=None,count=None,total=None):
    present={key for key in metadata if key.startswith('split.')}
    if not present:return None
    if present!=SPLIT_KEYS:raise ValueError('incomplete or unsupported GGUF split metadata')
    values=tuple(metadata[key] for key in ('split.no','split.count','split.tensors.count'))
    if (any(type(value) is not int for value in values) or not 2<=values[1]<=64
            or not 0<=values[0]<values[1] or not 1<=values[2]<=100_000
            or (index is not None and values!=(index,count,total))):
        raise ValueError('GGUF split index/count/tensor total disagrees or exceeds bounds')
    return values


@lru_cache(maxsize=2)
def _shard_paths(path,version):
    # This small cache retains names only, never token metadata or mmap views.
    with BoundedReader(path,allow_empty=True) as reader:
        split=split_metadata({name:field.contents() for name,field in reader.fields.items() if name.startswith('split.')})
        if split is None:
            if not reader.tensors:raise ValueError('single GGUF file has no tensors')
            paths=(path.resolve(),)
        else:
            index,count,_=split
            match=re.fullmatch(r'(.+)-00001-of-([0-9]{5})(\.gguf)',path.name,re.IGNORECASE)
            if index!=0 or match is None or int(match[2])!=count:
                raise ValueError('select the standard numbered first GGUF shard')
            paths=tuple(path.with_name(f'{match[1]}-{i:05d}-of-{count:05d}{match[3]}') for i in range(1,count+1))
    if identity(path)!=version:raise ValueError('GGUF file changed while discovering shards')
    return paths


def plan(path):
    path=Path(path).expanduser()
    # Resolve directories, but retain the file namespace containing sibling links.
    path=path.parent.resolve()/path.name
    paths=_shard_paths(path,identity(path))
    return _plan(paths[0],tuple((part,identity(part)) for part in paths))


@lru_cache(maxsize=2)
def _plan(path,version):
    metadata={};tensors={};headers=0;total=None
    for index,(part,stamp) in enumerate(version):
        with BoundedReader(part,allow_empty=index==0) as reader:
            headers+=int(reader.data_offset)
            if headers>MAX_HEADER:raise ValueError('aggregate GGUF shard metadata exceeds bounds')
            shard={name:field.contents() for name,field in reader.fields.items() if not name.startswith('GGUF.')}
            split=split_metadata(shard)
            if len(version)>1:
                if split is None:raise ValueError('GGUF shard is missing split metadata')
                if index==0:total=split[2]
                split_metadata(shard,index,len(version),total)
            elif split is not None:raise ValueError('GGUF split membership changed')
            shard={name:value for name,value in shard.items() if not name.startswith('split.')}
            if index==0:metadata=shard
            elif any(name!='general.alignment' and (name not in metadata or value!=metadata[name]) for name,value in shard.items()):
                raise ValueError('GGUF shard model/tokenizer metadata disagrees')
            for tensor in reader.tensors:
                if tensor.name in tensors:raise ValueError('duplicate tensor across GGUF shards')
                tensors[tensor.name]=TensorInfo(tensor.name,tuple(int(n) for n in reversed(tensor.shape)),
                    int(tensor.tensor_type),int(tensor.data_offset),int(tensor.n_bytes),index)
            if len(tensors)>100_000:raise ValueError('aggregate GGUF tensor count exceeds bounds')
        if identity(part)!=stamp:raise ValueError('GGUF shard changed while reading metadata')
    if total is not None and len(tensors)!=total:raise ValueError('GGUF split total tensor count disagrees')
    tok, special, eos = tokenizer(metadata)
    c = model_config(metadata, tensors, len(metadata['tokenizer.ggml.tokens']), eos)
    expected = set(); raw_groups = set()
    for destination, group in tensor_groups(c, tensors):
        kinds = set()
        for name, shape in group:
            tensor = tensors.get(name)
            if tensor is None or tensor.shape != shape:
                raise ValueError(f'GGUF tensor {name} is missing or has the wrong shape')
            if name == 'rope_freqs.weight' and tensor.kind != 0:
                raise ValueError('GGUF rotary factors require an F32 vector')
            expected.add(name)
            kinds.add(tensor.kind if tensor.kind in FORMATS else 0)
        if len(kinds) > 1 or kinds & KQUANTS:
            raw_groups.add(destination)
    if set(tensors) != expected:
        raise ValueError(f'unrecognized GGUF tensors: {sorted(set(tensors) - expected)[:4]}')
    kinds = {t.kind for t in tensors.values()} & FORMATS.keys()
    quantization = 'mixed' if len(kinds) > 1 else NAMES[next(iter(kinds))] if kinds else 'none'
    total = sum(t.host_bytes for t in tensors.values())
    embedding = tensors['token_embd.weight']
    if embedding.kind in (2, 8):
        # The Metal fallback expands legacy embeddings. CUDA uses the sum
        # of original tensor storage instead of this expanded estimate.
        total += math.prod(embedding.shape) * 2
        if not c['tie_word_embeddings']:
            total -= embedding.host_bytes
    template = metadata.get('tokenizer.chat_template')
    if template is not None and not isinstance(template, str):
        raise ValueError('GGUF needs a single text chat template')
    if any(identity(part)!=stamp for part,stamp in version):
        raise ValueError('GGUF file changed while reading metadata')
    return Plan(path, version, c, tensors, tok.to_str(), template, special, eos, quantization, total, frozenset(raw_groups),
                metadata.get('tokenizer.ggml.model') == 'llama' and metadata.get('tokenizer.ggml.add_space_prefix', True))


def weight_bytes(path, quantization=None):
    if mode(quantization) != 'none':
        raise ValueError('GGUF preserves checkpoint quantization; requested conversion is unsupported')
    source = plan(path)
    if source.raw_groups and selected_device() != 'cuda':
        raise ValueError('GGUF K-quants and mixed-format fused projections require CUDA')
    return (sum(t.host_bytes for t in source.tensors.values())
            if selected_device() == 'cuda' else source.host_bytes)


def read_tensor(file, tensor):
    file.seek(tensor.offset)
    raw = np.fromfile(file, dtype=np.uint8, count=tensor.length)
    if raw.size != tensor.length:
        raise ValueError(f'truncated GGUF tensor: {tensor.name}')
    if tensor.kind in (0, 1):
        dtype = '<f4' if tensor.kind == 0 else '<f2'
        target = torch.float32 if tensor.name == 'rope_freqs.weight' else torch.float16
        value = torch.from_numpy(raw.view(dtype).reshape(tensor.shape)).to(target)
        if not torch.isfinite(value).all():
            raise ValueError(f'GGUF tensor exceeds finite {target} range: {tensor.name}')
        if tensor.name == 'rope_freqs.weight' and not (value > 0).all():
            raise ValueError('GGUF rotary factors must be positive')
        return value, None
    block_bytes = 18 if tensor.kind == 2 else 34
    blocks = raw.reshape(-1, block_bytes)
    scales = torch.from_numpy(blocks[:, :2].copy().view('<f2').reshape(tensor.shape[0], -1))
    if not torch.isfinite(scales).all():
        raise ValueError(f'GGUF tensor has non-finite scales: {tensor.name}')
    packed = blocks[:, 2:].copy()
    packed = packed if tensor.kind == 2 else packed.view(np.int8)
    return torch.from_numpy(packed.reshape(tensor.shape[0], -1)), scales


def restore_llama_rows(name, part, config):
    """Invert GGUF's interleaved Q/K rotary rows, including packed block scales.

    The format mapping is documented by llama.cpp conversion/llama.py::permute.
    Qwen GGUF already uses the half-split row order consumed by our decoder.
    """
    if not name.endswith(('attn_q.weight', 'attn_k.weight')):
        return part
    heads = config['num_attention_heads' if name.endswith('attn_q.weight') else 'num_key_value_heads']
    def restore(value):
        if value is None:
            return None
        return value.reshape(heads, value.shape[0] // heads // 2, 2, *value.shape[1:]).transpose(1, 2).reshape(value.shape).contiguous()
    return tuple(restore(value) for value in part)


class GgufWeights(ModelWeights):
    def __init__(self, path, max_context, quantization=None):
        estimated = weight_bytes(path, quantization)
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
        from .gguf_sentencepiece import restore_tokenizer
        tokenizer = restore_tokenizer(source.tokenizer_json, source.sentencepiece_prefix)
        self.configure_tokenizer(tokenizer, source.template, source.special_tokens)
        self.eos = set(source.eos)
        weights, scales = {}, {}
        self.packed_formats = {}
        with ExitStack() as owned:
            files=[owned.enter_context(part.open('rb')) for part,_ in source.identity]
            if any(file_identity(os.fstat(file.fileno()))!=stamp for file,(_,stamp) in zip(files,source.identity)):
                raise ValueError('GGUF shard changed before payload loading')
            for destination, group in tensor_groups(self.config, source.tensors):
                if destination in source.raw_groups:
                    native = []
                    for name, shape in group:
                        info = source.tensors[name]
                        if info.kind in FORMATS:
                            value = read_raw(files[info.file_index], info, embedding=destination == 'embed')
                            kind = info.kind
                        else:
                            value, _ = read_tensor(files[info.file_index], info)
                            value = value.view(torch.uint8)
                            kind = 1  # Floating decoder weights preserve the FP16 policy.
                        if self.config['model_type'] in ('llama', 'mistral'):
                            value, _ = restore_llama_rows(name, (value, None), self.config)
                        native.append((value, kind))
                    self.packed_formats[destination] = packed_layout(group[0][1][1], native)
                    weights[destination] = torch.cat([value.reshape(-1) for value, _ in native])
                    continue
                parts = [read_tensor(files[source.tensors[name].file_index], source.tensors[name]) for name, _ in group]
                if self.config['model_type'] in ('llama', 'mistral'):
                    parts = [restore_llama_rows(name, part, self.config) for (name, _), part in zip(group, parts)]
                values = [p[0] for p in parts]
                weights[destination] = values[0] if len(values) == 1 else torch.cat(values)
                if parts[0][1] is not None:
                    parts_scales = [p[1] for p in parts]
                    scales[destination] = parts_scales[0] if len(parts_scales) == 1 else torch.cat(parts_scales)
            if any(file_identity(os.fstat(file.fileno()))!=stamp for file,(_,stamp) in zip(files,source.identity)):
                raise ValueError('GGUF shard changed while loading payloads')
        if self.config['tie_word_embeddings']:
            weights['head'] = weights['embed']
            if 'embed' in scales:
                scales['head'] = scales['embed']
            if 'embed' in self.packed_formats:
                self.packed_formats['head'] = self.packed_formats['embed']
        if 'embed' in scales:
            unpack = dequantize_q4 if weights['embed'].dtype == torch.uint8 else dequantize
            packed, step = weights['embed'], scales['embed']
            if selected_device() == 'cuda':
                # Preserve the existing FP16 lookup values without expanding the
                # vocabulary. Normal scales prove finiteness with a cheap bound;
                # extreme scales need a bounded exact check, including zero blocks.
                bound = 8 if packed.dtype == torch.uint8 else 128
                if step.abs().amax().item() > torch.finfo(torch.float16).max / bound:
                    for first in range(0, len(packed), 128):
                        values = unpack(packed[first:first+128], step[first:first+128], torch.float16)
                        if not torch.isfinite(values).all():
                            raise ValueError('GGUF embedding exceeds finite FP16 range')
            else:
                embedding = torch.empty(source.tensors['token_embd.weight'].shape, dtype=torch.float16)
                for first in range(0, len(embedding), 128):
                    embedding[first:first+128] = unpack(packed[first:first+128], step[first:first+128], torch.float16)
                if not torch.isfinite(embedding).all():
                    raise ValueError('GGUF embedding exceeds finite FP16 range')
                weights['embed'] = embedding
                del scales['embed']
        if any(identity(part)!=stamp for part,stamp in source.identity):
            raise ValueError('GGUF file changed while loading weights')
        self.finalize_weights(weights, scales)
        if self.bytes != estimated:
            raise ValueError('GGUF loaded storage disagrees with preflight estimate')
