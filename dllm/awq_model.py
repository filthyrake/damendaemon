"""Strict host import for AWQ GEMM; kept separate from runtime dispatch gates."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from .awq import Matrix, fuse, group_size
from .config import decoder_config, eos_ids, read_config
from .device import selected_device
from .download import read_index
from .qwen2 import ModelWeights, weight_files
from .quantization import mode


def config(path):
    raw = read_config(Path(path)/'config.json')
    group = group_size(raw.get('quantization_config'))
    architectures = {'qwen2': ['Qwen2ForCausalLM'], 'qwen3': ['Qwen3ForCausalLM']}
    if raw.get('model_type') not in architectures or raw.get('architectures') != architectures[raw['model_type']]:
        raise ValueError('AWQ requires a Qwen2 or Qwen3 dense decoder')
    if type(raw.get('tie_word_embeddings', False)) is not bool:
        raise ValueError('invalid AWQ tied-embedding declaration')
    c = decoder_config({k: v for k, v in raw.items() if k != 'quantization_config'})
    if c.get('rope_scaling'):
        raise ValueError('AWQ scaled rotary execution is not qualified')
    for _, projections in projection_groups(c):
        if any(inputs % group or outputs % 8 for _, inputs, outputs in projections):
            raise ValueError('AWQ projection dimensions must match group and channel packing')
    return {**c, 'weight_format': 'awq_gemm', 'awq_group_size': group}


def projection_groups(c):
    hidden, intermediate = c['hidden_size'], c['intermediate_size']
    query, key = c['num_attention_heads']*c['head_dim'], c['num_key_value_heads']*c['head_dim']
    for layer in range(c['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        yield f'{layer}.qkv.weight', [(prefix+'self_attn.'+name+'_proj', hidden, outputs)
                                     for name, outputs in (('q', query), ('k', key), ('v', key))]
        yield f'{layer}.out', [(prefix+'self_attn.o_proj', query, hidden)]
        yield f'{layer}.gate_up', [(prefix+'mlp.'+name+'_proj', hidden, intermediate) for name in ('gate', 'up')]
        yield f'{layer}.down', [(prefix+'mlp.down_proj', intermediate, hidden)]


def dense_groups(c):
    hidden = c['hidden_size']
    yield 'embed', [('model.embed_tokens.weight', (c['vocab_size'], hidden))]
    yield 'norm', [('model.norm.weight', (hidden,))]
    if not c.get('tie_word_embeddings'):
        yield 'head', [('lm_head.weight', (c['vocab_size'], hidden))]
    for layer in range(c['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        yield f'{layer}.norm1', [(prefix+'input_layernorm.weight', (hidden,))]
        yield f'{layer}.norm2', [(prefix+'post_attention_layernorm.weight', (hidden,))]
        if c['model_type'] == 'qwen2':
            yield f'{layer}.qkv.bias', [(prefix+'self_attn.'+name+'_proj.bias', (heads*c['head_dim'],))
                                       for name, heads in (('q', c['num_attention_heads']),
                                                           ('k', c['num_key_value_heads']), ('v', c['num_key_value_heads']))]
        if c['model_type'] == 'qwen3':
            for kind in ('q', 'k'):
                yield f'{layer}.{kind}_norm', [(prefix+f'self_attn.{kind}_norm.weight', (c['head_dim'],))]


def identity(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class Plan:
    path: Path
    config: dict
    tensors: dict
    identities: dict
    host_bytes: int

    def check_source(self):
        if any(identity(path) != expected for path, expected in self.identities.items()):
            raise ValueError('AWQ checkpoint changed during import')


def plan(path):
    path = Path(path).resolve()
    identities = {path/name: identity(path/name) for name in
                  ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                   'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja')}
    c = config(path)
    dense_dtypes = ('F16', 'BF16') if c['model_type'] == 'qwen3' else ('F16',)
    expected = {name: (shape, dense_dtypes) for _, entries in dense_groups(c) for name, shape in entries}
    if c.get('tie_word_embeddings'):
        expected['lm_head.weight'] = ((c['vocab_size'], c['hidden_size']), dense_dtypes)
    for _, entries in projection_groups(c):
        for prefix, inputs, outputs in entries:
            expected[prefix+'.qweight'] = ((inputs, outputs//8), ('I32',))
            expected[prefix+'.qzeros'] = ((inputs//128, outputs//8), ('I32',))
            expected[prefix+'.scales'] = ((inputs//128, outputs), ('F16',))
    tensors = {}
    try:
        files = weight_files(path)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('invalid AWQ shard index') from error
    for file in files:
        identities[file] = identity(file)
        with safe_open(str(file), framework='pt', device='cpu') as source:
            for name in source.keys():
                view = source.get_slice(name)
                signature = (tuple(view.get_shape()), view.get_dtype())
                if (name in tensors or name not in expected or signature[0] != expected[name][0]
                        or signature[1] not in expected[name][1]):
                    raise ValueError(f'unexpected AWQ tensor or schema: {name}')
                tensors[name] = (file, *signature)
    missing = set(expected) - tensors.keys()
    if c.get('tie_word_embeddings'):
        missing.discard('lm_head.weight')
    if missing:
        raise ValueError(f'missing AWQ tensors: {sorted(missing)}')
    index = path/'model.safetensors.index.json'
    if index.exists():
        declared = read_index(index).get('weight_map')
        actual = {name: str(file.relative_to(path)) for name, (file, _, _) in tensors.items()}
        if declared != actual:
            raise ValueError('AWQ shard index does not match tensor locations')
    total = sum(math.prod(shape)*(4 if dtype == 'I32' else 2)
                for name, (_, shape, dtype) in tensors.items()
                if not (name == 'lm_head.weight' and c.get('tie_word_embeddings')))
    result = Plan(path, c, tensors, identities, total)
    result.check_source()
    return result


def weight_bytes(path, quantization=None):
    if mode(quantization) != 'none':
        raise ValueError('AWQ checkpoint weights cannot be requantized on load')
    if selected_device() != 'cuda':
        raise ValueError('AWQ execution currently requires CUDA')
    return plan(path).host_bytes


class AwqWeights(ModelWeights):
    def __init__(self, path, max_context, quantization=None):
        if mode(quantization) != 'none' or selected_device() != 'cuda':
            raise ValueError('AWQ requires CUDA and checkpoint-native quantization')
        prepared = plan(path)
        self.path, self.config = prepared.path, prepared.config
        c = self.config
        if type(max_context) is not int or not 1 <= max_context <= c['max_position_embeddings']:
            raise ValueError('context exceeds AWQ model position limit')
        self.max_context = max_context
        self.quantization, self.weight_format = 'none', 'awq_gemm'
        self.weight_cache_status = 'disabled'
        tokenizer = Tokenizer.from_file(str(self.path/'tokenizer.json'))
        tokenizer_config = json.loads((self.path/'tokenizer_config.json').read_text())
        template = tokenizer_config.get('chat_template')
        if template is None and (self.path/'chat_template.jinja').exists():
            template = (self.path/'chat_template.jinja').read_text()
        self.configure_tokenizer(tokenizer, template, tokenizer_config)
        self.eos = eos_ids(c.get('eos_token_id'))
        if (self.path/'generation_config.json').exists():
            self.eos.update(eos_ids(json.loads((self.path/'generation_config.json').read_text()).get('eos_token_id')))
        if not self.eos or any(token >= c['vocab_size'] for token in self.eos):
            raise ValueError('invalid AWQ EOS token IDs')
        vocabulary = tokenizer.get_vocab()
        if not vocabulary or max(vocabulary.values()) >= c['vocab_size']:
            raise ValueError('AWQ tokenizer exceeds the embedding vocabulary')
        if any(tokenizer.id_to_token(token) is None for token in self.eos):
            raise ValueError('AWQ EOS token is missing from the tokenizer')
        state = {}
        for file in dict.fromkeys(item[0] for item in prepared.tensors.values()):
            with safe_open(str(file), framework='pt', device='cpu') as source:
                for name in source.keys():
                    state[name] = source.get_tensor(name)
        prepared.check_source()
        self.weights, self.scales, self.awq_matrices = {}, {}, {}
        for target, entries in dense_groups(c):
            values = [state.pop(name) for name, _ in entries]
            if any(not bool(torch.isfinite(value).all()) for value in values):
                raise ValueError(f'nonfinite AWQ dense tensor: {target}')
            if any(value.dtype == torch.bfloat16 and float(value.abs().max()) > torch.finfo(torch.float16).max
                   for value in values):
                raise ValueError(f'AWQ dense tensor exceeds FP16 execution range: {target}')
            self.weights[target] = values[0] if len(values) == 1 else torch.cat(values)
        if c.get('tie_word_embeddings'):
            head = state.pop('lm_head.weight', None)
            if head is not None and not torch.equal(head, self.weights['embed']):
                raise ValueError('declared AWQ tied head differs from embedding')
            self.weights['head'] = self.weights['embed']
        for target, entries in projection_groups(c):
            matrices = [Matrix(*(state.pop(prefix+suffix) for suffix in ('.qweight', '.qzeros', '.scales')),
                               inputs, outputs) for prefix, inputs, outputs in entries]
            matrix = matrices[0] if len(matrices) == 1 else fuse(matrices)
            if float(matrix.scales.max()) > torch.finfo(torch.float16).max/15:
                raise ValueError('AWQ coefficient range exceeds the qualified FP16 bound')
            self.awq_matrices[target] = matrix
            self.weights[target], self.scales[target] = matrix.weight, matrix.scales
        if state:
            raise ValueError(f'unconsumed AWQ tensors: {sorted(state)}')
        self.bytes = sum(t.numel()*t.element_size() for t in
                         {id(t): t for t in (*self.weights.values(), *self.scales.values(),
                                            *(m.zeros for m in self.awq_matrices.values()))}.values())
        if self.bytes != prepared.host_bytes:
            raise ValueError('AWQ prepared storage disagrees with admission estimate')
        prepared.check_source()
