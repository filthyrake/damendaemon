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


def stage_budget(weights, config, context, batch, cache_element_bytes, start, end, fp32_weights=False,
                 scales=None):
    names = stage_weight_names(weights, config['num_hidden_layers'], start, end)
    selected = {name: weights[name] for name in names}
    unique = {id(t):t for t in selected.values()}.values()
    weight_bytes = sum(t.numel()*(4 if fp32_weights else t.element_size()) for t in unique)
    scale_bytes = 0
    if scales:
        selected_names = set(names)
        scale_bytes = sum(t.numel()*(4 if fp32_weights else t.element_size())
                          for name, t in scales.items() if name in selected_names)
    owned = {**config, 'num_hidden_layers': end-start, 'layer_types': config['layer_types'][start:end]}
    state_bytes = hybrid_state_bytes(owned, context, batch, cache_element_bytes)
    head = selected.get('head')
    head_workspace = head.numel()*4 if head is not None and not fp32_weights and head.element_size() != 4 else 0
    result = dict(layer_start=start, layer_end=end, weight_bytes=weight_bytes, scale_bytes=scale_bytes,
                  state_bytes=state_bytes,
                  activation_workspace_bytes=2*batch*context*config['hidden_size']*4,
                  output_head_workspace_bytes=head_workspace, workspace_floor_bytes=512*1024**2)
    result['required_memory_bytes'] = sum(result[name] for name in (
        'weight_bytes', 'scale_bytes', 'state_bytes', 'activation_workspace_bytes',
        'output_head_workspace_bytes', 'workspace_floor_bytes'))
    return result


# Per-rank weight formats admitted by the mixed-precision planner: the two
# packed layouts quantization.prepare builds plus unconverted checkpoints.
STAGE_FORMATS = ('none', 'q8_0', 'q4_0')


def logical_shapes(weights, scales=None):
    """Format-independent logical shapes for every stage destination name.

    Packed q4_0 weights arrive as (rows, width//2) uint8 and q8_0 as
    (rows, width) int8; both carry (rows, width//32) float16 scales, so the
    logical width is scale columns * 32 and the packed width must agree.
    Everything else (embeddings, norms) reports its stored shape unchanged.
    """
    result = {}
    for name, tensor in weights.items():
        scale = (scales or {}).get(name)
        if scale is None or tensor.ndim != 2:
            result[name] = tuple(tensor.shape)
            continue
        if scale.ndim != 2 or tensor.ndim != 2:
            raise ValueError(f'packed tensor {name} has non-matrix scales')
        width = scale.shape[1] * 32
        if (tensor.shape[0] != scale.shape[0]
                or tensor.shape[1] not in (width // 2, width)):
            raise ValueError(f'packed tensor {name} disagrees with its scales')
        result[name] = (tensor.shape[0], width)
    return result


def stage_pricing_table(weights, scales=None, config=None):
    """Canonical per-stage pricing input extracted from one real load.

    The table is format-independent (logical shapes plus stored element
    sizes for non-matrix tensors and the tied embedding/head relationship
    from the model config), so every rank of a mixed-format chain derives
    the identical table from its own load and the deterministic split
    stays reproducible across formats.
    """
    shapes = logical_shapes(weights, scales)
    # Only non-matrix element sizes are recorded: matrix and embedding data
    # bytes come from the per-format rules, so a packed load's uint8/int8
    # element sizes never leak into the table and every format derives the
    # identical pricing input.
    elements = {name: tensor.element_size() for name, tensor in weights.items()
                if tensor.ndim != 2}
    groups = {}
    for name, tensor in weights.items():
        groups.setdefault(id(tensor), []).append(name)
    shared = set(tuple(sorted(names)) for names in groups.values() if len(names) > 1)
    # Canonicalize the tied embedding/head pair: a raw load shares one
    # tensor, a packed load rebinds the head, but the model's tie is a
    # config property — record it identically for every extraction format.
    if (config is not None and config.get('tie_word_embeddings')
            and 'embed' in weights and 'head' in weights):
        shared.add(('embed', 'head'))
    shared_storage = sorted(shared)
    scale_groups = {}
    if scales:
        for name, tensor in scales.items():
            scale_groups.setdefault(id(tensor), []).append(name)
    shared_scales = sorted(tuple(sorted(names)) for names in scale_groups.values() if len(names) > 1)
    return dict(shapes=shapes, element_bytes=elements, shared_storage=shared_storage,
                shared_scales=shared_scales)


def _format_tensor_bytes(name, shape, element_bytes, fmt):
    """Storage DATA bytes for one destination name under a packed format.

    Scales are accounted separately (stage_budget keeps packed data and
    scales in disjoint dicts). quantization.prepare packs two-dimensional
    non-embedding matrices — a tied output head included; the embedding
    itself always loads as raw float16. Unconverted matrices are float16.
    """
    if fmt not in STAGE_FORMATS:
        raise ValueError(f'unknown per-rank weight format: {fmt!r}')
    if len(shape) == 2 and name != 'embed':
        rows, width = shape
        if width % 32:
            raise ValueError(f'stage matrix {name} width is not 32-aligned')
        if fmt == 'none':
            return 2 * rows * width
        return (rows * width) // (2 if fmt == 'q4_0' else 1)
    total = 1
    for extent in shape:
        total *= extent
    if name == 'embed':
        return 2 * total
    # Non-matrix tensors (norms, conv state) stay as loaded.
    return element_bytes * total


def _dedup_shared(names, shared, config=None, fmt=None, *, split_names=True):
    """Resolve shared-storage names for a stage range under a target format.

    A tied embedding shares raw storage with the output head: under the
    unquantized format the pair counts once; under a packed format the head
    is rebound to its own packed tensor, so both count (raw embedding plus
    packed head). Any OTHER shared storage split across a stage boundary is
    rejected — it cannot move as two stages.
    """
    covered = set()
    for group in shared:
        present = [name for name in group if name in names]
        if not present:
            continue
        if (config is not None and fmt is not None
                and set(group) == {'embed', 'head'} and config.get('tie_word_embeddings')
                and 'embed' in names and 'head' in names):
            if fmt == 'none':
                covered.add('head')
            continue
        if len(present) == len(group):
            if split_names:
                covered.update(present[1:])
        elif len(present) > 1:
            raise ValueError(f'shared storage {group} crosses a stage range')
    return {name for name in names if name not in covered}


def stage_budget_formats(pricing, config, context, batch, cache_element_bytes, start, end, fmt):
    """stage_budget for a hypothetical per-rank format, priced by rules.

    Weight/scale bytes follow the packed-layout conventions above; state and
    workspace accounting is shared with the tensor-based stage_budget. Tests
    pin the rule predictions to real loads in every admitted format.
    """
    shapes = pricing['shapes']
    elements = pricing['element_bytes']
    names = set(stage_weight_names(shapes, config['num_hidden_layers'], start, end))
    weight_names = _dedup_shared(names, pricing['shared_storage'], config, fmt)
    quantized = {name for name in names if name in shapes and len(shapes[name]) == 2
                 and name != 'embed' and shapes[name][1] % 32 == 0}
    scale_names = _dedup_shared(quantized, pricing['shared_scales'], config, fmt,
                                split_names=False)
    weight_bytes = sum(_format_tensor_bytes(name, shapes[name], elements.get(name, 2), fmt)
                       for name in weight_names)
    scale_bytes = 0
    if fmt != 'none':
        scale_bytes = sum(2 * shapes[name][0] * (shapes[name][1] // 32)
                          for name in scale_names)
    owned = {**config, 'num_hidden_layers': end-start, 'layer_types': config['layer_types'][start:end]}
    state_bytes = hybrid_state_bytes(owned, context, batch, cache_element_bytes)
    head = shapes.get('head')
    head_workspace = 0
    if head is not None and 'head' in names and len(head) == 2 and head[1] % 32 == 0:
        packed_numel = (head[0] * head[1]) // (2 if fmt == 'q4_0' else 1)
        head_workspace = packed_numel * 4
    activation_workspace = 2*batch*context*config['hidden_size']*4
    required = (weight_bytes + scale_bytes + state_bytes + activation_workspace
                + head_workspace + 512*1024**2)
    return dict(layer_start=start, layer_end=end, weight_bytes=weight_bytes,
                scale_bytes=scale_bytes, state_bytes=state_bytes,
                activation_workspace_bytes=activation_workspace,
                output_head_workspace_bytes=head_workspace,
                workspace_floor_bytes=512*1024**2,
                required_memory_bytes=required, weight_format=fmt)


def choose_stage_splits(weights, config, context, batch, cache_element_bytes, budgets, fp32_weights=False,
                        scales=None):
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
                                                   start, end, fp32_weights, scales)
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


def choose_stage_splits_formats(pricing, config, context, batch, cache_element_bytes, budgets, formats):
    """Per-rank-format variant of choose_stage_splits (#601).

    budgets[i] and formats[i] describe rank i; each rank's stage is priced
    in that rank's format from the format-independent pricing table, so
    every rank derives the identical deterministic split while the chain
    carries mixed precision. A uniform format list reduces to the same
    search as choose_stage_splits with that format's costs.
    """
    from fractions import Fraction
    count = config['num_hidden_layers']
    if (not isinstance(budgets, (list, tuple)) or not isinstance(formats, (list, tuple))
            or len(budgets) != len(formats)
            or not 1 <= len(budgets) <= min(16, count)
            or any(type(value) is not int or value <= 0 for value in budgets)
            or any(value not in STAGE_FORMATS for value in formats)
            or count > 256):
        raise ValueError('mixed-format planning requires matching 1..16 budgets and formats')
    candidates = {}
    states = {0: (Fraction(0), ())}
    for rank, (limit, fmt) in enumerate(zip(budgets, formats)):
        next_states = {}
        last = count-(len(budgets)-rank-1)
        for start, (score, ranges) in states.items():
            for end in range(start+1, last+1):
                key = (start, end, fmt)
                if key not in candidates:
                    candidates[key] = stage_budget_formats(pricing, config, context, batch,
                                                            cache_element_bytes, start, end, fmt)
                required = candidates[key]['required_memory_bytes']
                if required > limit:
                    continue
                candidate = (max(score, Fraction(required, limit)), (*ranges, key))
                if end not in next_states or candidate < next_states[end]:
                    next_states[end] = candidate
        states = next_states
    if count not in states:
        return dict(storage_plan_found=False, device_budget_bytes=list(budgets),
                    per_rank_formats=list(formats),
                    reason='No contiguous nonempty partition fits the supplied storage budgets '
                           'at the requested per-rank formats')
    score, ranges = states[count]
    return dict(storage_plan_found=True, splits=[key[1] for key in ranges[:-1]],
                maximum_estimated_utilization=float(score),
                device_budget_bytes=list(budgets), per_rank_formats=list(formats),
                stages=[dict(candidates[key], device_budget_bytes=limit, weight_format=fmt,
                             remaining_budget_bytes=limit-candidates[key]['required_memory_bytes'])
                        for key, (limit, fmt) in zip(ranges, zip(budgets, formats))])
