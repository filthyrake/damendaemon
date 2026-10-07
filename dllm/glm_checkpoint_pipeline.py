"""Globally budgeted construction of a local packed GLM stage pipeline."""
from pathlib import Path
import stat
from .checkpoint import snapshot_shards
from .glm_reference_loader import _load_reference, load_reference_stage
from .glm_layer_stage import validate_range
from .glm_layer_pipeline import GlmLayerPipeline


def _inventory(path):
    files, index = snapshot_shards(path)
    if len(files) > 256: raise ValueError('GLM pipeline checkpoint exceeds shard limit')
    names = [path/'config.json', *files]
    if index is not None: names.append(path/'model.safetensors.index.json')
    result = []
    for file in names:
        info = file.stat()
        if not stat.S_ISREG(info.st_mode): raise ValueError('GLM pipeline requires regular checkpoint files')
        result.append((str(file), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


def load_reference_pipeline(path, layer_splits, *, max_weight_bytes=256*1024**2,
                            projection_workspace_bytes=8*1024**2,
                            max_workspace_bytes=64*1024**2, max_output_bytes=64*1024**2):
    """Audit the whole snapshot before reading any stage's tensor payloads.

    The global weight allowance includes all retained stage tensors plus the
    largest source conversion buffer. Stat receipts detect ordinary concurrent
    replacement/modification, not hostile writers or authenticated model bytes.
    """
    if (not isinstance(layer_splits, (list, tuple)) or not 1 <= len(layer_splits) <= 7
            or any(type(value) is not int or not 1 <= value < 128 for value in layer_splits)
            or list(layer_splits) != sorted(set(layer_splits))):
        raise ValueError('GLM pipeline needs 1..7 ordered distinct internal layer splits')
    path = Path(path).expanduser().resolve()
    before = _inventory(path)
    options = dict(max_weight_bytes=max_weight_bytes, projection_workspace_bytes=projection_workspace_bytes,
                   max_workspace_bytes=max_workspace_bytes, max_output_bytes=max_output_bytes)
    config, retained, overlap = _load_reference(path, None, inspect_only=True, **options)
    boundaries = (0, *layer_splits, config['num_hidden_layers'])
    ranges = tuple(zip(boundaries, boundaries[1:]))
    for start, end in ranges: validate_range(start, end, config['num_hidden_layers'], config['indexer_types'])
    if _inventory(path) != before: raise ValueError('GLM pipeline checkpoint changed during admission')
    stages = []
    try:
        for start, end in ranges:
            if _inventory(path) != before: raise ValueError('GLM pipeline checkpoint changed before stage loading')
            stage = load_reference_stage(path, start, end, **options)
            if stage.config != config or _inventory(path) != before:
                raise ValueError('GLM pipeline checkpoint changed during stage loading')
            stages.append(stage)
        pipeline = GlmLayerPipeline(stages)
        if _inventory(path) != before: raise ValueError('GLM pipeline checkpoint changed during construction')
    except BaseException:
        stages.clear()
        stage = None
        raise
    pipeline.retained_weight_bytes = retained
    pipeline.weight_overlap_bytes = overlap
    return pipeline
