"""Validate source tensors before decoder packing or cache publication."""


def validate(state, config):
    import torch
    from .floating_parameter import require_floating
    h=config['hidden_size'];d=config['head_dim']
    q=config['num_attention_heads']*d;kv=config['num_key_value_heads']*d
    inter=config['intermediate_size'];kind=config['model_type']
    shapes={'model.embed_tokens.weight':(config['vocab_size'],h),'model.norm.weight':(h,)}
    tied=config.get('tie_word_embeddings',False)
    if not tied or 'lm_head.weight' in state:shapes['lm_head.weight']=(config['vocab_size'],h)
    for layer in range(config['num_hidden_layers']):
        prefix=f'model.layers.{layer}.'
        shapes[prefix+'input_layernorm.weight']=(h,)
        shapes[prefix+'post_attention_layernorm.weight']=(h,)
        shapes[prefix+'self_attn.o_proj.weight']=(h,q)
        if kind=='olmoe':
            shapes[prefix+'mlp.gate.weight']=(config['num_experts'],h)
            for expert in range(config['num_experts']):
                root=prefix+f'mlp.experts.{expert}.'
                for projection in ('gate','up'):shapes[root+projection+'_proj.weight']=(inter,h)
                shapes[root+'down_proj.weight']=(h,inter)
        else:shapes[prefix+'mlp.down_proj.weight']=(h,inter)
        if kind=='phi3':
            shapes[prefix+'self_attn.qkv_proj.weight']=(q+2*kv,h)
            shapes[prefix+'mlp.gate_up_proj.weight']=(2*inter,h)
        else:
            for projection,width in (('q',q),('k',kv),('v',kv)):
                shapes[prefix+f'self_attn.{projection}_proj.weight']=(width,h)
                if kind=='qwen2':shapes[prefix+f'self_attn.{projection}_proj.bias']=(width,)
            if kind!='olmoe':
                for projection in ('gate','up'):shapes[prefix+f'mlp.{projection}_proj.weight']=(inter,h)
        if kind in ('qwen3','gemma3_text'):
            for projection in ('q','k'):shapes[prefix+f'self_attn.{projection}_norm.weight']=(d,)
        if kind=='gemma3_text':
            for name in ('pre_feedforward_layernorm','post_feedforward_layernorm'):
                shapes[prefix+name+'.weight']=(h,)
        if kind=='olmoe':
            shapes[prefix+'self_attn.q_norm.weight']=(q,)
            shapes[prefix+'self_attn.k_norm.weight']=(kv,)
    missing=set(shapes)-set(state)
    if missing:raise ValueError('missing dense checkpoint tensor: '+min(missing))
    extra=set(state)-set(shapes)
    if extra:raise ValueError('unexpected dense checkpoint tensor: '+min(extra))
    for name,shape in shapes.items():
        require_floating(state[name],name)
        if tuple(state[name].shape)!=shape:
            raise ValueError('dense checkpoint shape mismatch: '+name)
    if tied and 'lm_head.weight' in state and not torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight']):
        raise ValueError('dense checkpoint declares unequal tied output weights')
