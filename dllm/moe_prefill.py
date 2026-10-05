"""Conservative selection for independently qualified OLMoE grouped prefill."""


def qualified_scope(config, device_name, quantization, context, max_batch):
    geometry = dict(model_type='olmoe', hidden_size=2048, intermediate_size=1024,
                    num_hidden_layers=16, num_attention_heads=16,
                    num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                    vocab_size=50304, norm_topk_prob=False)
    return (device_name == 'NVIDIA A2' and quantization == 'q8_0'
            and 1 <= context <= 2048 and 1 <= max_batch <= 4
            and all(config.get(key) == value for key, value in geometry.items())
            and config.get('clip_qkv') is None)


def use_grouped_prefill(qualified, tokens):
    # Longer prompts retain stable host grouping: the device path regressed
    # at 256/512/1024 tokens on the qualification GPU.
    return qualified and 1 <= tokens <= 128
