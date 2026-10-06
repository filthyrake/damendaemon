"""Shared safetensors header inventory; no tensor values or device allocation."""
import json
from pathlib import Path

from safetensors import safe_open, SafetensorError

from .download import INDEX, snapshot_files


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate checkpoint index field')
        result[name] = value
    return result


def inspect_snapshot(path):
    path = Path(path)
    index = (json.loads((path/INDEX).read_text(), object_pairs_hook=_unique_fields)
             if (path/INDEX).exists() else None)
    names = [entry.name for entry in path.iterdir() if entry.is_file()]
    # Validate indexed paths before examining them, including nested shards.
    if isinstance(index, dict) and isinstance(index.get('weight_map'), dict):
        names.extend(name for name in index['weight_map'].values() if isinstance(name, str))
    selected = snapshot_files(names, index)
    files = [name for name in selected if name.endswith('.safetensors')]
    if any(not (path/name).is_file() for name in files):
        raise ValueError('indexed safetensors shard is missing')
    actual, metadata = {}, {}
    for filename in files:
        tensors = {}
        try:
            with safe_open(str(path/filename), framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name in actual:
                        raise ValueError(f'duplicate tensor: {name}')
                    header = reader.get_slice(name)
                    tensors[name] = {'shape':header.get_shape(), 'dtype':header.get_dtype()}
                    actual[name] = filename
        except SafetensorError as error:
            raise ValueError('invalid safetensors header') from error
        metadata[filename] = {'tensors':tensors}
    if index is not None and actual != index['weight_map']:
        raise ValueError('tensor headers do not match the shard index')
    return [path/name for name in files], metadata
