"""Validate supported dense decoder variants before loading weight tensors."""

def decoder_config(config):
    c = dict(config)
    kind = c.get('model_type')
    if kind not in ('qwen2', 'qwen3', 'llama'):
        raise ValueError(f'unsupported model architecture: {kind}; expected qwen2, qwen3, or llama')
    if (c.get('quantization_config') or c.get('rope_scaling') or c.get('rope_parameters')
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
    heads = c['num_attention_heads']
    if heads % c['num_key_value_heads']:
        raise ValueError('query head count must be divisible by KV head count')
    if 'head_dim' not in c and c['hidden_size'] % heads:
        raise ValueError('hidden size must divide evenly into heads without an explicit head_dim')
    c.setdefault('head_dim', c['hidden_size'] // heads)
    if type(c['head_dim']) is not int or c['head_dim'] <= 0 or c['head_dim'] % 2:
        raise ValueError('head_dim must be a positive even integer')
    c.setdefault('rope_theta', 10000.0)
    if c['rope_theta'] <= 0 or c.get('rms_norm_eps', 0) <= 0:
        raise ValueError('RoPE base and normalization epsilon must be positive')
    return c
