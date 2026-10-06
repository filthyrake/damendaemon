"""Shared, allocation-free hybrid state and stage storage accounting."""

def hybrid_state_bytes(config, context, batch_size, cache_element_bytes):
    if (type(context) is not int or context <= 0
            or type(batch_size) is not int or not 1 <= batch_size <= 16
            or type(cache_element_bytes) is not int or cache_element_bytes not in (2, 4)):
        raise ValueError('invalid hybrid state geometry')
    full = config['layer_types'].count('full_attention')
    linear = config['num_hidden_layers']-full
    recurrent = config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim']*4
    conv_width = (2*config['linear_num_key_heads']*config['linear_key_head_dim']
                  + config['linear_num_value_heads']*config['linear_value_head_dim'])
    return batch_size*(full*2*config['num_key_value_heads']*context*config['head_dim']*cache_element_bytes
                       + linear*(recurrent+conv_width*config['linear_conv_kernel_dim']*cache_element_bytes))



def stage_weight_names(weights, layer_count, start, end):
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= layer_count):
        raise ValueError('invalid hybrid stage range')
    selected = []
    present = set()
    for name in weights:
        prefix, separator, _ = name.partition('.')
        if separator and prefix.isascii() and prefix.isdecimal():
            index = int(prefix)
            if not 0 <= index < layer_count:
                raise ValueError('hybrid weight layer exceeds model range')
            if start <= index < end:
                selected.append(name)
                present.add(index)
        elif name == 'embed' and start == 0:
            selected.append(name)
        elif name in ('norm', 'head') and end == layer_count:
            selected.append(name)
        elif name not in ('embed', 'norm', 'head'):
            raise ValueError(f'unrecognized hybrid stage weight: {name}')
    if present != set(range(start, end)):
        raise ValueError('missing hybrid stage layer weights')
    required = ({'embed'} if start == 0 else set()) | ({'norm', 'head'} if end == layer_count else set())
    if not required <= set(selected):
        raise ValueError('missing hybrid stage boundary weights')
    return tuple(selected)


def stage_budget(weights, config, context, batch, cache_element_bytes, start, end, fp32_weights=False):
    names = stage_weight_names(weights, config['num_hidden_layers'], start, end)
    selected = {name: weights[name] for name in names}
    unique = {id(t):t for t in selected.values()}.values()
    weight_bytes = sum(t.numel()*(4 if fp32_weights else t.element_size()) for t in unique)
    owned = {**config, 'num_hidden_layers': end-start, 'layer_types': config['layer_types'][start:end]}
    state_bytes = hybrid_state_bytes(owned, context, batch, cache_element_bytes)
    head = selected.get('head')
    head_workspace = head.numel()*4 if head is not None and not fp32_weights and head.element_size() != 4 else 0
    result = dict(layer_start=start, layer_end=end, weight_bytes=weight_bytes, state_bytes=state_bytes,
                  activation_workspace_bytes=2*batch*context*config['hidden_size']*4,
                  output_head_workspace_bytes=head_workspace, workspace_floor_bytes=512*1024**2)
    result['required_memory_bytes'] = sum(result[name] for name in (
        'weight_bytes', 'state_bytes', 'activation_workspace_bytes', 'output_head_workspace_bytes', 'workspace_floor_bytes'))
    return result
