"""Header-only hybrid storage planning; not execution admission or qualification."""
from .config import read_config
from pathlib import Path


from .hybrid_layout import hybrid_state_bytes


def estimate(inventory, metadata, cache_dtype='float16', contexts=(32768, 65536, 131072, 262144), batches=(1, 4), layer_splits=None, stage_weight_dtype='native', device_budgets=None):
    import torch
    from .config import decoder_config
    from .hybrid import load_weights
    if cache_dtype not in ('float16', 'float32'):
        raise ValueError('cache dtype must be float16 or float32')
    if stage_weight_dtype not in ('native', 'float32') or (layer_splits is None and device_budgets is None and stage_weight_dtype != 'native'):
        raise ValueError('stage weight dtype requires stage splits or budgets and must be native or float32')
    if layer_splits is not None and device_budgets is not None:
        raise ValueError('choose explicit stage splits or device budgets')
    cache_element_bytes = 2 if cache_dtype == 'float16' else 4
    config = decoder_config(inventory['config'])
    if config['model_type'] != 'qwen3_5_text':
        raise ValueError('capacity planner requires a supported hybrid text configuration')
    if (not contexts or not batches
            or any(type(c) is not int or not 1 <= c <= config['max_position_embeddings'] for c in contexts)
            or any(type(b) is not int or not 1 <= b <= 16 for b in batches)):
        raise ValueError('contexts must fit model positions; batch sizes must be 1..16')
    if layer_splits is not None:
        if (not isinstance(layer_splits, (list, tuple))
                or any(type(i) is not int or not 0 < i < config['num_hidden_layers'] for i in layer_splits)
                or list(layer_splits) != sorted(set(layer_splits))):
            raise ValueError('layer splits must be strictly increasing internal layer boundaries')
    state = {}
    for shard in metadata['files_metadata'].values():
        for name, tensor in shard['tensors'].items():
            if name in state:
                raise ValueError(f'duplicate tensor: {name}')
            state[name] = torch.empty(tensor['shape'], device='meta')
    try:
        weights = load_weights(state, config)
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError('hybrid tensor header geometry does not match the text weight mapping') from error
    h, inter = config['hidden_size'], config['intermediate_size']
    dim, qh, kh = config['head_dim'], config['num_attention_heads'], config['num_key_value_heads']
    lk, lv = config['linear_num_key_heads'], config['linear_num_value_heads']
    dk, dv = config['linear_key_head_dim'], config['linear_value_head_dim']
    conv = 2*lk*dk+lv*dv
    expected = {'embed': (config['vocab_size'], h), 'head': (config['vocab_size'], h), 'norm': (h,)}
    for i, kind in enumerate(config['layer_types']):
        shapes = {'norm1': (h,), 'norm2': (h,), 'gate_up': (2*inter, h), 'down': (h, inter)}
        if kind == 'full_attention':
            shapes.update(qkv=(2*qh*dim+2*kh*dim, h), o=(h, qh*dim), q_norm=(dim,), k_norm=(dim,))
        else:
            shapes.update(qkv=(conv, h), z=(lv*dv, h), b=(lv, h), a=(lv, h),
                          conv=(conv, 1, config['linear_conv_kernel_dim']), out=(h, lv*dv),
                          dt=(lv,), A=(lv,), gated_norm=(dv,))
        expected.update({f'{i}.{name}': shape for name, shape in shapes.items()})
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f'hybrid tensor geometry disagrees with config: {name}')
    unexpected = [name for name in state if not name.startswith(('mtp.', 'model.visual.'))]
    if unexpected:
        raise ValueError(f'unconsumed text tensors: {unexpected}')
    totals = {}
    for mode in ('none', 'q8_0', 'q4_0'):
        total = 0
        seen = set()
        for name, tensor in weights.items():
            packed = mode != 'none' and tensor.ndim == 2 and name != 'embed'
            identity = (id(tensor), packed)
            if identity in seen:
                continue
            seen.add(identity)
            n = tensor.numel()
            if packed:
                if tensor.shape[1] % 32:
                    raise ValueError(f'non-packable matrix: {name}')
                total += (n if mode == 'q8_0' else n // 2) + n // 32 * 2
            else:
                total += n * tensor.element_size()
        totals[mode] = total
    from .layer_stream import plan as plan_streaming
    streaming = plan_streaming(weights, config['num_hidden_layers'])
    head_workspace = weights['head'].numel()*4 if weights['head'].dtype != torch.float32 else 0
    rows = {}
    streamed_rows = {}
    for context in contexts:
        rows[str(context)] = {}
        streamed_rows[str(context)] = {}
        for users in batches:
            state_bytes = hybrid_state_bytes(config, context, users, cache_element_bytes)
            streamed_rows[str(context)][str(users)] = {
                'weights_state_and_head_workspace_bytes': streaming.peak_bytes+state_bytes+head_workspace}
            rows[str(context)][str(users)] = {
                'state_bytes': state_bytes,
                'weights_plus_state_bytes': {mode: size+state_bytes for mode, size in totals.items()},
            }
    result = dict(repository=inventory['repository'], revision=inventory['resolved_revision'],
                kv_and_convolution_dtype=cache_dtype, recurrent_dtype='float32',
                weight_bytes=totals, estimated_storage_formats=list(totals),
                layer_streaming=dict(weight_format='none', fixed_weight_bytes=streaming.fixed_bytes,
                    per_layer_weight_bytes=list(streaming.layer_bytes),
                    peak_resident_weight_bytes=streaming.peak_bytes,
                    output_head_workspace_bytes=head_workspace, contexts=streamed_rows,
                    limitation='Storage estimate only; active implementation has separate CUDA, FP16, eager and context limits.'),
                runtime_qualification='not assessed', contexts=rows, mapped_tensors=len(weights),
                excluded_tensors=len(state),
                limitation='Shape-only text-loader and packed-storage estimate. Excludes workspace, allocator, graphs, communication, host copies and loading peaks. No weight values, inference, quality or performance validated.')

    if layer_splits is not None:
        from .hybrid_layout import stage_budget
        edges = [0, *layer_splits, config['num_hidden_layers']]
        result['pipeline_stages'] = dict(splits=list(layer_splits), weight_dtype=stage_weight_dtype,
            weight_format='none', runtime_qualification='not assessed',
            contexts={str(context): {str(batch): [stage_budget(weights, config, context, batch,
                cache_element_bytes, start, end, fp32_weights=stage_weight_dtype == 'float32')
                for start, end in zip(edges, edges[1:])] for batch in batches} for context in contexts},
            limitation='Per-stage storage and workspace estimates only; excludes allocator fragmentation, loading peaks and network transport. No device fit or throughput qualification.')
    if device_budgets is not None:
        from .hybrid_layout import choose_stage_splits
        result['pipeline_placement'] = dict(weight_dtype=stage_weight_dtype, weight_format='none',
            runtime_qualification='not assessed',
            contexts={str(context): {str(batch): choose_stage_splits(weights, config, context, batch,
                cache_element_bytes, device_budgets, stage_weight_dtype == 'float32') for batch in batches}
                for context in contexts},
            limitation='Balances estimated storage across ordered input budgets, not latency or throughput. No live memory query or runtime qualification.')
    return result


def plan_local(path, contexts, batches, cache_dtype, layer_splits=None, stage_weight_dtype='native', device_budgets=None):
    """Inspect local safetensors headers only; never resolve or download a model."""
    from .checkpoint import inspect_snapshot
    from .config import decoder_config
    path = Path(path).expanduser().resolve()
    original = read_config(path/'config.json')
    if decoder_config(original)['model_type'] != 'qwen3_5_text':
        raise ValueError('capacity planner requires a supported hybrid text configuration')
    _, files = inspect_snapshot(path)
    if any(tensor['dtype'] not in ('F16', 'BF16', 'F32')
           for shard in files.values() for tensor in shard['tensors'].values()):
        raise ValueError('planner supports floating-point source checkpoints only')
    result = estimate({'config': original, 'repository': None, 'resolved_revision': None},
                      {'files_metadata': files}, cache_dtype, contexts, batches, layer_splits, stage_weight_dtype, device_budgets)
    result.update(local_path=str(path), source='local safetensors headers; contents not hashed',
                  workspace_admission_floor_bytes=512*1024**2)
    # The runtime's floor is admission bookkeeping, not a workspace peak bound.
    for users in result['contexts'].values():
        for row in users.values():
            row['weights_state_and_workspace_floor_bytes'] = {
                mode: size+result['workspace_admission_floor_bytes']
                for mode, size in row['weights_plus_state_bytes'].items()}
    for users in result['layer_streaming']['contexts'].values():
        for row in users.values():
            row['with_workspace_floor_bytes'] = (
                row['weights_state_and_head_workspace_bytes'] + result['workspace_admission_floor_bytes'])
    return result
