"""Select the HF checkpoint consumed by dllm, excluding alternate weight copies."""
from pathlib import PurePosixPath
import re


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
                    or any(c in name for c in '\\*?[]\x00') or not name.endswith('.safetensors')):
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


def gguf_file(filenames, filename):
    """Select one exact repository file, never an entire GGUF variant family."""
    if not isinstance(filename, str):
        raise ValueError('GGUF filename must be a repository-relative path')
    path = PurePosixPath(filename)
    if (path.is_absolute() or any(part in ('', '.', '..') for part in filename.split('/'))
            or any(c in filename for c in '\\*?[]\x00') or not filename.lower().endswith('.gguf')):
        raise ValueError('GGUF filename must be an exact repository-relative .gguf path')
    if filename not in set(filenames):
        raise ValueError(f'GGUF file missing from repository: {filename}')
    return filename


def gguf_files(filenames, filename):
    """Select one exact file or its bounded, complete standard shard family."""
    files = set(filenames)
    filename = gguf_file(files, filename)
    match = re.fullmatch(r'(.+)-([0-9]{5})-of-([0-9]{5})(\.gguf)', filename, re.IGNORECASE)
    if match is None:
        return (filename,)
    index, count = int(match[2]), int(match[3])
    if index != 1 or not 2 <= count <= 64:
        raise ValueError('select the first GGUF shard in a family of 2..64 files')
    names = tuple(f'{match[1]}-{i:05d}-of-{count:05d}{match[4]}' for i in range(1, count + 1))
    for name in names:
        gguf_file(files, name)
    return names
