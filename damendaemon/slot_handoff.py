"""Populated slot-state handoff between compatible stages (#606).

The payload side of the KV_HANDOFF protocol command: export a populated
slot's attention KV (plain or Q8_0-packed) plus conv/recurrent state as
bounded CPU tensors with a strict identity inventory, and install them
into a fresh slot of a compatible receiver stage — no recomputation, the
receiver's cursor commits only after the acknowledged round. A source
payload covering a SUPERSET layer range feeds per-rank receivers of a
layer-split pair (each rank imports its own slice).

No transport here: the caller moves the payload (in-process device copy
or the tensor channel's frames) and drives the protocol command. All
agreement checks are content-derived — config, quantization, KV mode,
compute dtype, context, batch, layer coverage and the full tensor
inventory — so a wrong model, a wrong precision or a stale inventory
refuses at install time, before any receiver state is touched.
"""
import time


def _clone(tensor):
    return tensor.detach().to('cpu').contiguous().clone()


def export_slot(stage, slot):
    """One populated slot's state as CPU tensors + identity inventory.

    Returns dict(metadata=..., tensors=...). Nothing is recorded on the
    stage: the sender keeps its slot until the caller releases it after
    the acknowledged handoff (release_slot). Names are group-tagged and
    carry the GLOBAL layer index: kv.data.<layer>, kv.scales.<layer>
    (Q8_0-packed caches only), conv.<layer>, state.<layer>.
    """
    started = time.perf_counter()
    if (getattr(stage, 'closed', False) or getattr(stage, 'failed', False)
            or type(slot) is not int or not 0 <= slot < stage.max_batch):
        raise ValueError('handoff export needs a live slot of this stage')
    position = stage.lengths[slot]
    if not 0 < position <= stage.context:
        raise ValueError('handoff export needs a populated slot')
    tensors = {}
    for layer in sorted(stage.kv):
        cache = stage.kv[layer]
        if isinstance(cache, tuple):
            data, scales = cache
            tensors[f'kv.data.{layer}'] = _clone(data[:, slot, :, :position, :])
            tensors[f'kv.scales.{layer}'] = _clone(scales[:, slot, :, :position, :])
        else:
            tensors[f'kv.data.{layer}'] = _clone(cache[:, slot, :, :position, :])
    for group in ('conv', 'state'):
        for layer in sorted(getattr(stage, group, {})):
            tensors[f'{group}.{layer}'] = _clone(getattr(stage, group)[layer][slot])
    metadata = dict(
        config=stage.c, quantization=stage.model.quantization,
        kv_mode=getattr(stage, 'kv_mode', 'none'),
        compute_dtype=str(stage.compute_dtype),
        layer_start=stage.layer_start, layer_end=stage.layer_end,
        context=stage.context, max_batch=stage.max_batch,
        position=position,
        inventory=[(name, str(tensor.dtype), list(tensor.shape),
                    tensor.numel() * tensor.element_size())
                   for name, tensor in sorted(tensors.items())])
    return dict(metadata=metadata, tensors=tensors,
                telemetry=dict(export_bytes=sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values()),
                    tensors=len(tensors),
                    export_seconds=time.perf_counter() - started))


def import_slot(stage, slot, generation, payload):
    """Install an exported payload into a FRESH slot of a compatible stage.

    Mirrors the protocol's stale-offset rejection at stage level: the target
    slot must be empty; the payload's layer range must cover this stage's;
    config/quantization/KV-mode/dtype/context/batch must agree; every
    inventory tensor must match the target slice's shape and dtype. The
    stage's cursor (lengths) commits only after every tensor is installed
    (the acknowledged-completion-before-commit rule, mirrored stage-side);
    a refused handoff raises before any receiver state is touched.
    """
    started = time.perf_counter()
    metadata = payload.get('metadata') or {}
    tensors = payload.get('tensors') or {}
    if (getattr(stage, 'closed', False) or getattr(stage, 'failed', False)
            or type(slot) is not int or not 0 <= slot < stage.max_batch
            or stage.lengths[slot]):
        raise ValueError('handoff import needs an empty slot of a live stage')
    if (metadata.get('config') != stage.c
            or metadata.get('quantization') != stage.model.quantization
            or metadata.get('kv_mode') != getattr(stage, 'kv_mode', 'none')
            or metadata.get('compute_dtype') != str(stage.compute_dtype)
            or metadata.get('context') != stage.context
            or metadata.get('max_batch') != stage.max_batch
            or not metadata.get('layer_start', 10 ** 9) <= stage.layer_start
            or not stage.layer_end <= metadata.get('layer_end', -1)
            or not 0 < metadata.get('position', 0) <= stage.context
            or not metadata.get('inventory')
            or sorted(tensors) != sorted(
                name for name, *_ in metadata.get('inventory', ()))):
        raise ValueError('handoff payload disagrees with this stage')
    position = metadata['position']
    installed = 0
    installed_count = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        parts = name.split('.')
        group = parts[0]
        layer = int(parts[-1])
        # A superset payload feeds each rank its own slice: entries outside
        # this stage's layer range belong to the other ranks.
        if not stage.layer_start <= layer < stage.layer_end:
            continue
        if group == 'kv':
            member = parts[1]
            cache = stage.kv[layer]
            if member == 'data':
                target = (cache[:, slot, :, :position, :] if not isinstance(cache, tuple)
                          else cache[0][:, slot, :, :position, :])
            elif member == 'scales' and isinstance(cache, tuple):
                target = cache[1][:, slot, :, :position, :]
            else:
                raise ValueError('handoff payload has an unknown KV member')
        elif group in ('conv', 'state'):
            target = getattr(stage, group)[layer][slot]
        else:
            raise ValueError('handoff payload has an unknown group')
        if target.shape != tensor.shape or target.dtype != tensor.dtype:
            raise ValueError(f'handoff payload shape/dtype disagrees for {name}')
        target.copy_(tensor.to(target.device))
        installed += tensor.numel() * tensor.element_size()
        installed_count += 1
    stage.lengths[slot] = position
    stage.length = stage.lengths[0]
    telemetry = dict(import_bytes=installed, tensors=installed_count,
                    import_seconds=time.perf_counter() - started,
                    slot=slot, generation=generation, position=position)
    stage.last_handoff = telemetry
    return telemetry


def release_slot(stage, slot):
    """Zero a slot's state and free its cursor (sender side, post-ack).

    Mirrors the worker RELEASE semantics: the full slot column is zeroed so
    the next prefill starts from a clean slot.
    """
    if (getattr(stage, 'closed', False)
            or type(slot) is not int or not 0 <= slot < stage.max_batch
            or not stage.lengths[slot]):
        raise ValueError('handoff release needs a populated slot')
    position = stage.lengths[slot]
    for cache in stage.kv.values():
        for member in (cache if isinstance(cache, tuple) else (cache,)):
            member[:, slot].zero_()
    for group in ('conv', 'state'):
        for tensor in getattr(stage, group, {}).values():
            tensor[slot].zero_()
    stage.lengths[slot] = 0
    stage.length = stage.lengths[0]
    return dict(slot=slot, released_position=position)
