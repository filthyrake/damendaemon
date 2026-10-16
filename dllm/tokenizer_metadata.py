"""Bounded checkpoint metadata; vocabulary tokenizer.json has a separate contract."""
from pathlib import Path
from .safetensor_header import parse_metadata_json

MAX_TOKENIZER_CONFIG_BYTES = 16 * 1024 * 1024
MAX_GENERATION_CONFIG_BYTES = 1024 * 1024


def _read_object(path, limit):
    path = Path(path)
    with path.open('rb') as reader:
        raw = reader.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'{path.name} exceeds metadata byte limit')
    value = parse_metadata_json(raw.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must be an object')
    return value


def read_tokenizer_config(path):
    return _read_object(path, MAX_TOKENIZER_CONFIG_BYTES)


def read_generation_config(path):
    return _read_object(path, MAX_GENERATION_CONFIG_BYTES)


def special_token_text(metadata):
    """Convert tokenizer AddedToken objects to the strings expected by Jinja."""
    result = {}
    for name in ('bos_token', 'eos_token'):
        value = metadata.get(name)
        if isinstance(value, dict):
            value = value.get('content')
            if not isinstance(value, str):
                raise ValueError(f'{name} metadata requires text content')
        elif value is not None and not isinstance(value, str):
            raise ValueError(f'{name} must be text or structured token metadata')
        result[name] = value
    return result
