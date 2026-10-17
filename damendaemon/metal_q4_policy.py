"""Narrow admission for independently qualified native Metal Q4 execution."""


def validate(config, context, max_batch=None):
    geometry = dict(model_type='qwen2', hidden_size=896, num_hidden_layers=24,
                    num_attention_heads=14, num_key_value_heads=2, intermediate_size=4864,
                    vocab_size=151936, head_dim=64)
    if (any(config.get(key) != value for key,value in geometry.items())
            or config.get('rope_scaling') or type(context) is not int or not 1 <= context <= 2048):
        raise ValueError('Metal decoder quantization Q4 requires Qwen2.5-0.5B geometry, standard RoPE and context <=2048')
    if max_batch is not None and (type(max_batch) is not int or not 1 <= max_batch <= 4):
        raise ValueError('Metal Q4 requires 1..4 request slots')
