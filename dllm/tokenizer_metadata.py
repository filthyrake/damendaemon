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


# Match the envelope for templates embedded in tokenizer_config.json.
MAX_CHECKPOINT_TEMPLATE_BYTES = MAX_TOKENIZER_CONFIG_BYTES


def _select_named_template(collection, *, tools=False):
    """Select an explicit HF chat variant without guessing from collection order."""
    if not 1 <= len(collection) <= 64:
        raise ValueError('named chat templates require 1..64 entries')
    if isinstance(collection, dict):
        entries = collection.items()
    else:
        if any(not isinstance(entry, dict) or set(entry) != {'name', 'template'}
               for entry in collection):
            raise ValueError('named chat templates require name/template entries')
        entries = ((entry['name'], entry['template']) for entry in collection)
    templates = {}
    size = 0
    for name, template in entries:
        if not isinstance(name, str) or not name or not isinstance(template, str):
            raise ValueError('named chat templates require nonempty names and text templates')
        if name in templates:
            raise ValueError('duplicate named chat template')
        try:
            size += len(template.encode('utf-8'))
            name.encode('utf-8')
        except UnicodeError as error:
            raise ValueError('named chat templates require valid UTF-8') from error
        if size > MAX_CHECKPOINT_TEMPLATE_BYTES:
            raise ValueError('named chat templates exceed metadata byte limit')
        templates[name] = template
    if 'default' not in templates:
        raise ValueError('named chat templates require an explicit default')
    return templates.get('tool_use', templates['default']) if tools else templates['default']


def checkpoint_template(path, metadata, *, tools=False):
    template = metadata.get('chat_template')
    if isinstance(template, (dict, list)):
        template = _select_named_template(template, tools=tools)
    if template is None:
        fallback = Path(path)/'chat_template.jinja'
        if fallback.exists():
            with fallback.open('rb') as reader:
                raw = reader.read(MAX_CHECKPOINT_TEMPLATE_BYTES + 1)
            if len(raw) > MAX_CHECKPOINT_TEMPLATE_BYTES:
                raise ValueError('checkpoint chat template exceeds metadata byte limit')
            template = raw.decode('utf-8')
    if template is not None and not isinstance(template, str):
        raise ValueError('model needs one text chat template')
    return template
