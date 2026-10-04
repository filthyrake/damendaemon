"""Validate supported dense decoder variants before loading weight tensors."""
import math


def rotary_scaling(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('rotary scaling must be an object')
    kind = value.get('rope_type', value.get('type'))
    if kind not in ('linear', 'llama3'):
        raise ValueError('supported rotary scaling types are linear and llama3')
    common = {'rope_type', 'type', 'factor'}
    extra = {'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings'}
    if set(value) - (common | (extra if kind == 'llama3' else set())):
        raise ValueError('unsupported rotary scaling fields')
    if 'type' in value and value['type'] != kind:
        raise ValueError('conflicting rotary scaling types')
    for name in ('factor',) + (('low_freq_factor', 'high_freq_factor') if kind == 'llama3' else ()):
        number = value.get(name)
        if type(number) not in (int,float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if value['factor'] < 1:
        raise ValueError('rotary scaling factor must be at least one')
    if kind == 'llama3':
        if value['high_freq_factor'] <= value['low_freq_factor']:
            raise ValueError('rotary frequency bounds must be ordered')
        original = value.get('original_max_position_embeddings')
        if type(original) is not int or original <= 0:
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
    c['rope_scaling'] = rotary_scaling(c.get('rope_scaling'))
    if c['rope_theta'] <= 0 or c.get('rms_norm_eps', 0) <= 0:
        raise ValueError('RoPE base and normalization epsilon must be positive')
    return c


def starcoder_config(config):
    c = dict(config)
    if (c.get('hidden_act') != 'gelu_pytorch_tanh' or type(c.get('use_bias', True)) is not bool
            or c.get('norm_type', 'layer_norm') != 'layer_norm' or c.get('mlp_type', 'default') != 'default'):
        raise ValueError('unsupported StarCoder2 activation or bias configuration')
    epsilon = c.get('norm_epsilon', 1e-5)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('LayerNorm epsilon must be finite and positive')
    common = decoder_config({**c, 'model_type': 'mistral', 'hidden_act': 'silu', 'rms_norm_eps': epsilon})
    return {**common, **c, 'head_dim': common['head_dim'], 'rms_norm_eps': epsilon,
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
    if clip is not None and (type(clip) not in (int,float) or not math.isfinite(clip) or clip<=0):
        raise ValueError('QKV clip must be finite and positive')
    if common['head_dim']*c['num_attention_heads'] != c['hidden_size']:
        raise ValueError('OLMoE requires whole-projection Q/K normalization dimensions')
    return {**common,**c,'head_dim':common['head_dim'],'norm_topk_prob':c.get('norm_topk_prob',False)}


def hybrid_config(config):
    """Validate the dense, text-only Qwen3.5 backbone before allocating weights."""
    c = dict(config.get('text_config', config))
    if c.get('model_type') != 'qwen3_5_text':
        raise ValueError('Qwen3.5 requires a text backbone configuration')
    if config.get('quantization_config') or c.get('quantization_config') or c.get('num_experts', 0):
        raise ValueError('quantized and MoE hybrid checkpoints are not supported')
    rope = c.get('rope_parameters', {})
    if (rope.get('rope_type') != 'default' or c.get('hidden_act') != 'silu'
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
    if len(layers) != c['num_hidden_layers'] or any(x not in ('linear_attention', 'full_attention') for x in layers):
        raise ValueError('invalid hybrid layer layout')
    if (c['num_attention_heads'] % c['num_key_value_heads']
            or c['linear_num_value_heads'] % c['linear_num_key_heads']):
        raise ValueError('query/value heads must divide into key heads')
    fraction = rope.get('partial_rotary_factor', 1.)
    dim = c['head_dim'] * fraction
    if not 0 < fraction <= 1 or int(dim) != dim or int(dim) % 2:
        raise ValueError('invalid partial rotary dimension')
    c['rotary_dim'] = int(dim)
    c['rope_theta'] = rope.get('rope_theta', 10000.)
    if c['rope_theta'] <= 0 or c.get('rms_norm_eps', 0) <= 0:
        raise ValueError('invalid rotary base or normalization epsilon')
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
    if (len(layers) != c.get('num_hidden_layers')
            or any(x not in ('sliding_attention', 'full_attention') for x in layers)):
        raise ValueError('Gemma requires an explicit valid layer layout')
    if type(c.get('sliding_window')) is not int or c['sliding_window'] <= 0:
        raise ValueError('Gemma sliding window must be a positive integer')
    for name in ('query_pre_attn_scalar', 'rope_local_base_freq'):
        value = c.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    # Share dimension and unsupported-weight-format checks with dense decoders.
    common = decoder_config({**c, 'model_type': 'llama', 'hidden_act': 'silu',
                             'layer_types': ['full_attention'] * len(layers)})
    return {**common, **c, 'head_dim': common['head_dim'],
            'tie_word_embeddings': c.get('tie_word_embeddings', True)}
