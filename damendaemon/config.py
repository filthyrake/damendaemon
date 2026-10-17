"""Validate supported dense decoder variants before loading weight tensors."""
import json
import math
from pathlib import Path
import sys

MAX_CONFIG_BYTES = 1024 * 1024


def read_config(path):
    """Read one bounded UTF-8 configuration object before tensor loading."""
    with Path(path).open('rb') as reader:
        raw = reader.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError('model configuration exceeds 1 MiB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ValueError('duplicate model configuration field')
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):raise ValueError('nonfinite model configuration number')
        return parsed
    infinity = object()
    infinity_count = 0
    def constant(value):
        nonlocal infinity_count
        if value != 'Infinity':raise ValueError('nonfinite model configuration number')
        infinity_count += 1
        return infinity
    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                            parse_float=number, parse_constant=constant)
    except RecursionError as error:
        raise ValueError('model configuration nesting is too deep') from error
    if not isinstance(result, dict):raise ValueError('model configuration must be an object')
    if infinity_count:
        # Transformers serializes Mamba-2's supported unbounded time-step
        # interval using the nonstandard JSON Infinity literal. Permit only
        # that exact field; a sentinel prevents accepting it anywhere else.
        limit = result.get('time_step_limit')
        if (infinity_count != 1 or result.get('model_type') != 'mamba2'
                or not isinstance(limit, list) or len(limit) != 2
                or type(limit[0]) not in (int, float) or limit[0] != 0
                or limit[1] is not infinity):
            raise ValueError('nonfinite model configuration number')
        result['time_step_limit'] = [limit[0], float('inf')]
    return result


def _positive_number(value):
    # Comparing first also handles JSON integers beyond the host float range.
    return type(value) in (int, float) and 0 < value <= sys.float_info.max


def _rotary_dimension(head_dim, fraction):
    if (type(head_dim) is not int or not _positive_number(head_dim)
            or type(fraction) not in (int,float) or not 0<fraction<=1):
        raise ValueError('invalid partial rotary geometry')
    width=head_dim*fraction
    result=int(width)
    if result!=width or result<2 or result%2:
        raise ValueError('invalid partial rotary dimension')
    return result


def rotary_scaling(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('rotary scaling must be an object')
    kind = value.get('rope_type', value.get('type'))
    if kind not in ('linear', 'llama3', 'yarn'):
        raise ValueError('supported rotary scaling types are linear, llama3 and yarn')
    common = {'rope_type', 'type', 'factor'}
    extra = {'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings'}
    yarn_fields = {'original_max_position_embeddings', 'attention_factor', 'beta_fast', 'beta_slow', 'truncate'}
    if set(value) - (common | (extra if kind == 'llama3' else yarn_fields if kind == 'yarn' else set())):
        raise ValueError('unsupported rotary scaling fields')
    if 'type' in value and value['type'] != kind:
        raise ValueError('conflicting rotary scaling types')
    for name in ('factor',) + (('low_freq_factor', 'high_freq_factor') if kind == 'llama3' else ()):
        number = value.get(name)
        if not _positive_number(number):
            raise ValueError(f'{name} must be finite and positive')
    if value['factor'] < 1:
        raise ValueError('rotary scaling factor must be at least one')
    if kind == 'llama3':
        if value['high_freq_factor'] <= value['low_freq_factor']:
            raise ValueError('rotary frequency bounds must be ordered')
        original = value.get('original_max_position_embeddings')
        if type(original) is not int or original <= 0:
            raise ValueError('original context must be a positive integer')
    if kind == 'yarn':
        if 'truncate' in value and type(value['truncate']) is not bool:
            raise ValueError('YaRN truncate must be boolean')
        for name in ('attention_factor', 'beta_fast', 'beta_slow'):
            number = value.get(name, 1)
            if not _positive_number(number):
                raise ValueError(f'{name} must be finite and positive')
        if value.get('beta_fast', 32) < value.get('beta_slow', 1):
            raise ValueError('YaRN fast rotation bound must not be below slow bound')
        original = value.get('original_max_position_embeddings')
        if 'original_max_position_embeddings' in value and (type(original) is not int or original <= 0):
            raise ValueError('original context must be a positive integer')
    return {**value, 'rope_type':kind}


def eos_ids(value):
    if value is None:
        return set()
    values = value if isinstance(value,list) else [value]
    if not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError('EOS must be a token ID or nonempty list of token IDs')
    return set(values)

def decoder_config(config):
    if not isinstance(config, dict):raise ValueError('decoder configuration must be an object')
    if config.get('model_type') == 'gpt_oss':
        from .gpt_oss import validate_config
        return validate_config(config)
    scaling = config.get('rope_scaling')
    if (isinstance(scaling, dict) and scaling.get('rope_type', scaling.get('type')) == 'yarn'
            and config.get('model_type') not in ('qwen2', 'qwen3', 'llama', 'mistral')):
        raise ValueError('YaRN is currently supported only for Qwen2/Qwen3/Llama/Mistral')
    if config.get('model_type') == 'bloom':
        return bloom_config(config)
    if config.get('model_type') == 'gpt_neox':
        return neox_config(config)
    if config.get('model_type') == 'gpt2':
        return gpt2_config(config)
    if config.get('model_type') == 'olmoe':
        return olmoe_config(config)
    if config.get('model_type') == 'starcoder2':
        return starcoder_config(config)
    if config.get('model_type') == 'gemma3_text':
        return gemma_config(config)
    if config.get('model_type') in ('qwen3_5', 'qwen3_5_text'):
        return hybrid_config(config)
    c = dict(config)
    kind = c.get('model_type')
    if kind not in ('qwen2', 'qwen3', 'llama', 'phi3', 'mistral'):
        raise ValueError(f'unsupported model architecture: {kind}; expected qwen2, qwen3, llama, phi3, or mistral')
    if (c.get('quantization_config') or c.get('rope_parameters')
            or c.get('use_sliding_window', False) or c.get('hidden_act') != 'silu'
            or c.get('partial_rotary_factor', 1) != 1 or c.get('mlp_bias', False)
            or not isinstance(c.get('layer_types', []), list)
            or any(layer != 'full_attention' for layer in c.get('layer_types', []))):
        raise ValueError('only dense SiLU decoders with full attention and standard full-head RoPE are supported')
    if kind != 'qwen2' and c.get('attention_bias', False):
        raise ValueError('biased attention is only supported for Qwen2')
    for name in ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                 'num_hidden_layers', 'max_position_embeddings', 'vocab_size'):
        if type(c.get(name)) is not int or c[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if kind in ('phi3', 'mistral') and c.get('sliding_window') is not None:
        if type(c['sliding_window']) is not int or c['sliding_window'] <= 0:
            raise ValueError('sliding_window must be a positive integer')
    heads = c['num_attention_heads']
    if heads % c['num_key_value_heads']:
        raise ValueError('query head count must be divisible by KV head count')
    if 'head_dim' not in c and c['hidden_size'] % heads:
        raise ValueError('hidden size must divide evenly into heads without an explicit head_dim')
    c.setdefault('head_dim', c['hidden_size'] // heads)
    if type(c['head_dim']) is not int or c['head_dim'] <= 0 or c['head_dim'] % 2:
        raise ValueError('head_dim must be a positive even integer')
    c.setdefault('rope_theta', 10000.0)
    for name in ('rope_theta', 'rms_norm_eps'):
        number = c.get(name)
        if not _positive_number(number):
            raise ValueError(f'{name} must be finite and positive')
    c['rope_scaling'] = rotary_scaling(c.get('rope_scaling'))
    scaling = c['rope_scaling']
    if scaling and scaling['rope_type'] == 'yarn':
        if c['rope_theta'] <= 1 or scaling.get('original_max_position_embeddings', 1) > c['max_position_embeddings']:
            raise ValueError('YaRN needs a base above one and an original context no larger than the maximum')
    if c['rope_theta'] <= 0 or c.get('rms_norm_eps', 0) <= 0:
        raise ValueError('RoPE base and normalization epsilon must be positive')
    return c


def starcoder_config(config):
    c = dict(config)
    if (c.get('hidden_act') != 'gelu_pytorch_tanh' or type(c.get('use_bias', True)) is not bool
            or c.get('norm_type', 'layer_norm') != 'layer_norm' or c.get('mlp_type', 'default') != 'default'):
        raise ValueError('unsupported StarCoder2 activation or bias configuration')
    epsilon = c.get('norm_epsilon', 1e-5)
    if not _positive_number(epsilon):
        raise ValueError('LayerNorm epsilon must be finite and positive')
    common = decoder_config({**c, 'model_type': 'mistral', 'hidden_act': 'silu', 'rms_norm_eps': epsilon})
    return {**common, **c, 'head_dim': common['head_dim'], 'rms_norm_eps': epsilon,
            'rope_scaling': common['rope_scaling'],
            'use_bias': c.get('use_bias', True), 'tie_word_embeddings': c.get('tie_word_embeddings', True)}


def olmoe_config(config):
    c = dict(config)
    c.setdefault('rms_norm_eps',1e-5)
    common = decoder_config({**c, 'model_type':'llama'})
    for name in ('num_experts','num_experts_per_tok'):
        if type(c.get(name)) is not int or c[name]<=0:
            raise ValueError('expert counts must be positive integers')
    if c['num_experts_per_tok']>c['num_experts'] or type(c.get('norm_topk_prob',False)) is not bool:
        raise ValueError('invalid expert routing configuration')
    clip = c.get('clip_qkv')
    if clip is not None and not _positive_number(clip):
        raise ValueError('QKV clip must be finite and positive')
    if common['head_dim']*c['num_attention_heads'] != c['hidden_size']:
        raise ValueError('OLMoE requires whole-projection Q/K normalization dimensions')
    return {**common,**c,'head_dim':common['head_dim'],'rope_scaling':common['rope_scaling'],
            'norm_topk_prob':c.get('norm_topk_prob',False)}


def hybrid_config(config):
    """Validate the dense, text-only Qwen3.5 backbone before allocating weights."""
    text = config.get('text_config', config)
    if not isinstance(text, dict):raise ValueError('hybrid text configuration must be an object')
    c = dict(text)
    if c.get('model_type') != 'qwen3_5_text':
        raise ValueError('Qwen3.5 requires a text backbone configuration')
    if config.get('quantization_config') or c.get('quantization_config') or c.get('num_experts', 0):
        raise ValueError('quantized and MoE hybrid checkpoints are not supported')
    rope = c.get('rope_parameters', {})
    if (not isinstance(rope, dict) or rope.get('rope_type') != 'default' or c.get('hidden_act') != 'silu'
            or c.get('attention_bias', False) or c.get('mlp_bias', False)
            or not c.get('attn_output_gate', False)):
        raise ValueError('unsupported hybrid attention, activation, or rotary configuration')
    for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                'num_key_value_heads', 'head_dim', 'vocab_size', 'max_position_embeddings',
                'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
                'linear_value_head_dim', 'linear_conv_kernel_dim'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    layers = c.get('layer_types', [])
    if not isinstance(layers, list) or len(layers) != c['num_hidden_layers'] or any(x not in ('linear_attention', 'full_attention') for x in layers):
        raise ValueError('invalid hybrid layer layout')
    if (c['num_attention_heads'] % c['num_key_value_heads']
            or c['linear_num_value_heads'] % c['linear_num_key_heads']):
        raise ValueError('query/value heads must divide into key heads')
    fraction = rope.get('partial_rotary_factor', 1.)
    c['rotary_dim'] = _rotary_dimension(c['head_dim'], fraction)
    c['rope_theta'] = rope.get('rope_theta', 10000.)
    for name in ('rope_theta', 'rms_norm_eps'):
        number = c.get(name)
        # Compare before conversion: JSON integers can exceed the float range.
        if not _positive_number(number):
            raise ValueError(f'{name} must be finite and positive')
    c['tie_word_embeddings'] = config.get('tie_word_embeddings', c.get('tie_word_embeddings', False))
    c['multimodal_token_ids'] = [config[k] for k in ('image_token_id', 'video_token_id', 'vision_start_token_id',
                                                   'vision_end_token_id') if k in config]
    return c


def gemma_config(config):
    """Gemma 3 text uses explicit local/global attention and four layer norms."""
    c = dict(config)
    if (c.get('hidden_activation') != 'gelu_pytorch_tanh'
            or c.get('attn_logit_softcapping') is not None
            or c.get('final_logit_softcapping') is not None
            or c.get('use_bidirectional_attention', False)
            or c.get('attention_bias', False) or c.get('rope_scaling') is not None):
        raise ValueError('unsupported Gemma activation, attention, softcap, or rotary scaling')
    layers = c.get('layer_types', [])
    if (not isinstance(layers, list) or len(layers) != c.get('num_hidden_layers')
            or any(x not in ('sliding_attention', 'full_attention') for x in layers)):
        raise ValueError('Gemma requires an explicit valid layer layout')
    if type(c.get('sliding_window')) is not int or c['sliding_window'] <= 0:
        raise ValueError('Gemma sliding window must be a positive integer')
    for name in ('query_pre_attn_scalar', 'rope_local_base_freq'):
        value = c.get(name)
        if not _positive_number(value):
            raise ValueError(f'{name} must be finite and positive')
    # Share dimension and unsupported-weight-format checks with dense decoders.
    common = decoder_config({**c, 'model_type': 'llama', 'hidden_act': 'silu',
                             'layer_types': ['full_attention'] * len(layers)})
    return {**common, **c, 'head_dim': common['head_dim'],
            'tie_word_embeddings': c.get('tie_word_embeddings', True)}


def gpt2_config(config):
    c=dict(config)
    if (c.get('architectures') != ['GPT2LMHeadModel'] or c.get('activation_function') != 'gelu_new'
            or c.get('add_cross_attention',False) or c.get('scale_attn_weights',True) is not True
            or c.get('scale_attn_by_inverse_layer_idx',False) or c.get('reorder_and_upcast_attn',False)
            or c.get('tie_word_embeddings',True) is not True or c.get('quantization_config')):
        raise ValueError('unsupported GPT-2 activation, attention or checkpoint format')
    for name in ('n_embd','n_head','n_layer','n_positions','vocab_size'):
        if type(c.get(name)) is not int or c[name]<=0:
            raise ValueError('invalid GPT-2 dimensions')
    if c['n_embd'] % c['n_head'] or (c['n_embd']//c['n_head']) % 2:
        raise ValueError('unsupported GPT-2 head geometry')
    inter=c.get('n_inner')
    inter=4*c['n_embd'] if inter is None else inter
    epsilon=c.get('layer_norm_epsilon',1e-5)
    if type(inter) is not int or inter<=0 or not _positive_number(epsilon):
        raise ValueError('invalid GPT-2 MLP or normalization')
    return {**c,'hidden_size':c['n_embd'],'intermediate_size':inter,
            'num_attention_heads':c['n_head'],'num_key_value_heads':c['n_head'],
            'num_hidden_layers':c['n_layer'],'head_dim':c['n_embd']//c['n_head'],
            'max_position_embeddings':c['n_positions'],'rms_norm_eps':epsilon,
            'tie_word_embeddings':True,'rope_theta':10000.0}


def neox_config(config):
    c = dict(config)
    if (c.get('architectures') != ['GPTNeoXForCausalLM'] or c.get('hidden_act') != 'gelu'
            or c.get('rope_scaling') is not None or c.get('rope_parameters') is not None
            or c.get('tie_word_embeddings',False) is not False or c.get('attention_bias',True) is not True
            or c.get('quantization_config') or type(c.get('use_parallel_residual',True)) is not bool):
        raise ValueError('unsupported GPT-NeoX activation, attention or weight format')
    for name in ('hidden_size','intermediate_size','num_attention_heads','num_hidden_layers','max_position_embeddings','vocab_size'):
        if type(c.get(name)) is not int or c[name]<=0:
            raise ValueError('invalid GPT-NeoX dimensions')
    if c['hidden_size'] % c['num_attention_heads']:
        raise ValueError('GPT-NeoX hidden size must divide into heads')
    dim = c['hidden_size']//c['num_attention_heads']
    fraction = c.get('rotary_pct',.25)
    base, epsilon = c.get('rotary_emb_base',10000), c.get('layer_norm_eps',1e-5)
    rotary_dim = _rotary_dimension(dim, fraction)
    for value in (base,epsilon):
        if not _positive_number(value):
            raise ValueError('invalid GPT-NeoX rotary base or LayerNorm epsilon')
    return {**c,'head_dim':dim,'num_key_value_heads':c['num_attention_heads'],
            'rotary_dim':rotary_dim,'rope_theta':base,'rope_scaling':None,
            'rms_norm_eps':epsilon,'use_parallel_residual':c.get('use_parallel_residual',True)}


def bloom_config(config):
    c = dict(config)
    h = c.get('hidden_size', c.get('n_embed'))
    heads = c.get('n_head', c.get('num_attention_heads'))
    layers = c.get('n_layer', c.get('num_hidden_layers'))
    context = c.get('max_position_embeddings', 2048)
    for left, right in (('hidden_size', 'n_embed'), ('n_head', 'num_attention_heads'), ('n_layer', 'num_hidden_layers')):
        if left in c and right in c and c[left] != c[right]:
            raise ValueError('conflicting BLOOM dimension aliases')
    if (c.get('architectures') != ['BloomForCausalLM'] or c.get('quantization_config')
            or c.get('tie_word_embeddings', True) is not True
            or type(c.get('pretraining_tp', 1)) is not int or c.get('pretraining_tp', 1) != 1
            or c.get('slow_but_exact', False)
            or c.get('rope_scaling') or c.get('rope_parameters')
            or type(c.get('apply_residual_connection_post_layernorm', False)) is not bool):
        raise ValueError('unsupported BLOOM configuration or checkpoint format')
    for value in (h, heads, layers, context, c.get('vocab_size')):
        if type(value) is not int or value <= 0:
            raise ValueError('invalid BLOOM dimensions')
    if h % heads or (h // heads) % 2 or c.get('n_inner') not in (None, 4*h):
        raise ValueError('unsupported BLOOM head or MLP geometry')
    epsilon = c.get('layer_norm_epsilon', 1e-5)
    if not _positive_number(epsilon):
        raise ValueError('invalid BLOOM LayerNorm epsilon')
    return {**c, 'hidden_size': h, 'num_attention_heads': heads, 'num_key_value_heads': heads,
            'num_hidden_layers': layers, 'head_dim': h // heads, 'intermediate_size': 4*h,
            'max_position_embeddings': context, 'rms_norm_eps': epsilon,
            'tie_word_embeddings': True, 'rope_theta': 10000.0, 'rope_scaling': None,
            'apply_residual_connection_post_layernorm': c.get('apply_residual_connection_post_layernorm', False)}
