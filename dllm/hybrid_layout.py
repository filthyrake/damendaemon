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


def choose_stage_splits(weights, config, context, batch, cache_element_bytes, budgets, fp32_weights=False):
    """Minimize maximum estimated utilization across ordered device budgets.

    Contiguous, nonempty stages only. This balances storage, not transfer or
    compute time; the supplied budgets are inputs, never live device queries.
    """
    from fractions import Fraction
    count = config['num_hidden_layers']
    if (not isinstance(budgets, (list, tuple)) or not 1 <= len(budgets) <= min(16, count)
            or any(type(value) is not int or value <= 0 for value in budgets)
            or count > 256):
        raise ValueError('stage planning requires 1..16 positive byte budgets, at least one layer per stage and at most 256 layers')
    # (range -> shared admission budget) avoids recomputing tensor accounting
    # for the same candidate on different participating devices.
    candidates = {}
    states = {0: (Fraction(0), ())}
    for rank, limit in enumerate(budgets):
        next_states = {}
        last = count-(len(budgets)-rank-1)
        for start, (score, ranges) in states.items():
            for end in range(start+1, last+1):
                key = (start, end)
                if key not in candidates:
                    candidates[key] = stage_budget(weights, config, context, batch, cache_element_bytes,
                                                   start, end, fp32_weights)
                required = candidates[key]['required_memory_bytes']
                if required > limit:
                    continue
                candidate = (max(score, Fraction(required, limit)), (*ranges, key))
                if end not in next_states or candidate < next_states[end]:
                    next_states[end] = candidate
        states = next_states
    if count not in states:
        return dict(storage_plan_found=False, device_budget_bytes=list(budgets),
                    reason='No contiguous nonempty partition fits the supplied storage budgets')
    score, ranges = states[count]
    return dict(storage_plan_found=True, splits=[end for _, end in ranges[:-1]],
                maximum_estimated_utilization=float(score), device_budget_bytes=list(budgets),
                stages=[dict(candidates[key], device_budget_bytes=limit,
                             remaining_budget_bytes=limit-candidates[key]['required_memory_bytes'])
                        for key, limit in zip(ranges, budgets)])
