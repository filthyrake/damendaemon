"""Select the HF checkpoint consumed by dllm, excluding alternate weight copies."""
from pathlib import PurePosixPath


INDEX = 'model.safetensors.index.json'


def snapshot_files(filenames, index=None):
    files=set(filenames)
    if INDEX in files:
        if not isinstance(index,dict) or not isinstance(index.get('weight_map'),dict) or not index['weight_map']:
            raise ValueError('checkpoint index needs a nonempty weight_map')
        weights=set()
        for tensor,name in index['weight_map'].items():
            if not isinstance(tensor,str) or not tensor or not isinstance(name,str):
                raise ValueError('checkpoint index needs string tensor names and shard paths')
            path=PurePosixPath(name)
            if (path.is_absolute() or any(part in ('','.','..') for part in name.split('/'))
                    or any(c in name for c in '\\*?[]') or not name.endswith('.safetensors')):
                raise ValueError('invalid checkpoint shard path')
            if name not in files:raise ValueError(f'checkpoint shard missing from repository: {name}')
            weights.add(name)
    elif 'model.safetensors' in files:
        weights={'model.safetensors'}
    else:
        raise ValueError('repository needs a root model.safetensors or model.safetensors.index.json')
    # Preserve the previously supported JSON/template metadata, including nested
    # pooling configs. Only weights are restricted to the loader's checkpoint.
    metadata={name for name in files if name.endswith(('.json','.jinja'))
              or PurePosixPath(name).name.startswith('LICENSE') or name=='README.md'}
    # snapshot_download interprets allow_patterns as globs, even for exact names.
    if any(any(c in name for c in '*?[]') for name in metadata):
        raise ValueError('repository metadata paths cannot contain glob metacharacters')
    return sorted(weights|metadata)
