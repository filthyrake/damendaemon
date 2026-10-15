"""Shared safetensors header inventory; no tensor values or device allocation."""
from pathlib import Path

from safetensors import safe_open, SafetensorError

from .download import INDEX, snapshot_files, read_index
from .safetensor_header import read_header



def snapshot_shards(path):
    """Validate bounded index metadata and shard paths without opening shards."""
    path = Path(path)
    index = None
    if (path/INDEX).exists():
        index = read_index(path/INDEX)
    names = [entry.name for entry in path.iterdir() if entry.is_file()]
    # Validate indexed paths before examining them, including nested shards.
    if isinstance(index, dict) and isinstance(index.get('weight_map'), dict):
        names.extend(name for name in index['weight_map'].values() if isinstance(name, str))
    selected = snapshot_files(names, index)
    files = [name for name in selected if name.endswith('.safetensors')]
    if any(not (path/name).is_file() for name in files):
        raise ValueError('indexed safetensors shard is missing')
    return [path/name for name in files], index


def inspect_snapshot(path):
    path = Path(path)
    files, index = snapshot_shards(path)
    actual, metadata = {}, {}
    for file in files:
        filename = str(file.relative_to(path))
        tensors = {}
        try:
            with safe_open(str(path/filename), framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    header = reader.get_slice(name)
                    tensors[name] = {'shape':header.get_shape(), 'dtype':header.get_dtype()}
        except SafetensorError as error:
            # Older safetensors releases reject E8M0 before exposing any slices.
            # Only this explicit byte format gets the independently validated
            # header fallback; payload values remain unread.
            try:
                with (path/filename).open('rb') as raw_reader:
                    entries, _, _ = read_header(raw_reader)
                if not any(entry['dtype'] in ('F8_E8M0','F8_E8M0FNU') for entry in entries.values()):
                    raise ValueError('no E8M0 compatibility fallback applies')
                tensors = {name:{'shape':entry['shape'],'dtype':entry['dtype']}
                           for name,entry in entries.items()}
            except ValueError:
                raise ValueError('invalid safetensors header') from error
        for name in tensors:
            if name in actual:
                raise ValueError(f'duplicate tensor: {name}')
            actual[name] = filename
        metadata[filename] = {'tensors':tensors}
    if index is not None and actual != index['weight_map']:
        raise ValueError('tensor headers do not match the shard index')
    return files, metadata
