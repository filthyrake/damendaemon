"""Hybrid source tensor validation shared by loading and offline meta planning."""


def validate(state,c,prefix):
    import torch
    h=c['hidden_size'];inter=c['intermediate_size'];d=c['head_dim']
    q=c['num_attention_heads']*d;kv=c['num_key_value_heads']*d
    lk=c['linear_num_key_heads'];lv=c['linear_num_value_heads']
    dk=c['linear_key_head_dim'];dv=c['linear_value_head_dim']
    shapes={prefix+'embed_tokens.weight':(c['vocab_size'],h),prefix+'norm.weight':(h,)}
    if not c['tie_word_embeddings'] or 'lm_head.weight' in state:shapes['lm_head.weight']=(c['vocab_size'],h)
    for layer,kind in enumerate(c['layer_types']):
        root=prefix+f'layers.{layer}.'
        for name in ('input_layernorm','post_attention_layernorm'):shapes[root+name+'.weight']=(h,)
        for name in ('gate','up'):shapes[root+f'mlp.{name}_proj.weight']=(inter,h)
        shapes[root+'mlp.down_proj.weight']=(h,inter)
        if kind=='full_attention':
            for name,width in (('q',2*q),('k',kv),('v',kv)):
                shapes[root+f'self_attn.{name}_proj.weight']=(width,h)
            shapes[root+'self_attn.o_proj.weight']=(h,q)
            for name in ('q','k'):shapes[root+f'self_attn.{name}_norm.weight']=(d,)
        else:
            width=2*lk*dk+lv*dv
            for name,rows in (('qkv',width),('z',lv*dv),('b',lv),('a',lv)):
                shapes[root+f'linear_attn.in_proj_{name}.weight']=(rows,h)
            shapes[root+'linear_attn.out_proj.weight']=(h,lv*dv)
            shapes[root+'linear_attn.conv1d.weight']=(width,1,c['linear_conv_kernel_dim'])
            for name in ('dt_bias','A_log'):shapes[root+'linear_attn.'+name]=(lv,)
            shapes[root+'linear_attn.norm.weight']=(dv,)
    missing=set(shapes)-set(state)
    if missing:raise ValueError('missing hybrid checkpoint tensor: '+min(missing))
    extra={name for name in state if name not in shapes and not name.startswith(('mtp.','model.visual.'))}
    if extra:raise ValueError('unexpected hybrid checkpoint tensor: '+min(extra))
    for name,shape in shapes.items():
        if tuple(state[name].shape)!=shape:raise ValueError('hybrid checkpoint shape mismatch: '+name)
    if c['tie_word_embeddings'] and 'lm_head.weight' in state:
        head,embed=state['lm_head.weight'],state[prefix+'embed_tokens.weight']
        # Offline capacity planning has shapes but no values to compare.
        if head.device.type==embed.device.type=='meta':return
        if head.device.type=='meta' or embed.device.type=='meta' or not torch.equal(head,embed):
            raise ValueError('hybrid checkpoint declares unequal tied output weights')
