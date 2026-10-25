"""Compatibility and two-phase metadata publication for resident stage weights.

The owner must fence GPU work before copying and synchronize every participating
stage before committing metadata. A copy failure invalidates the whole owner;
these helpers cannot roll back partially overwritten device buffers.
"""
import copy

from .layer_stage import shard_model


def detached_model(model):
    """Clone a model's tensors so no stage construction aliases its weights.

    CPU stage construction reuses model tensors without copying (.to() on a
    matching CPU tensor is identity), so an in-place switch copy would corrupt
    a source the live stage shares. Switch sources must be private. Ties
    within weights (and within scales) are preserved.
    """
    detached = copy.copy(model)
    seen = {}

    def private(tensor):
        return seen.setdefault(id(tensor), tensor.clone())

    detached.weights = {name: private(tensor) for name, tensor in model.weights.items()}
    detached.scales = {name: private(tensor) for name, tensor in model.scales.items()}
    return detached


def compatible_shard(stage, model):
    """Return a checked host shard, or None, without modifying resident state."""
    try:
        shard = shard_model(model, *stage.layer_range)
    except ValueError:
        return None
    if (stage.model.config != shard.config
            or stage.model.quantization != shard.quantization
            or stage.model.max_context != shard.max_context
            or stage.w.keys() != shard.weights.keys()
            or stage.model.scales.keys() != shard.scales.keys()):
        return None

    def aliases(values):
        first = {}
        return {name: first.setdefault(id(value), name) for name, value in values.items()}

    if aliases(stage.w) != aliases(shard.weights):
        return None
    scales = {name: stage.quant_linear.scales[id(stage.w[name])] for name in shard.scales}
    if aliases(scales) != aliases(shard.scales):
        return None
    for targets, sources in ((stage.w, shard.weights), (scales, shard.scales)):
        for name, target in targets.items():
            source = sources[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                return None
    return shard


def copy_stage_weights(stage, shard):
    """Update stable device addresses; leave model identity unpublished."""
    for name, target in stage.w.items():
        target.copy_(shard.weights[name])
    for name, source in shard.scales.items():
        stage.quant_linear.scales[id(stage.w[name])].copy_(source)
    if stage.q4_linear is not None:
        stage.q4_linear.refresh()


def commit_stage_binding(stage, shard, max_batch):
    """Publish only after the owner's complete local/peer success fence."""
    stage.model = shard
    stage.c = shard.config
    stage.lengths = [0] * max_batch
