"""Shared sandboxed chat templates and explicit per-model overrides."""
from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

MAX_TEMPLATE_BYTES = 128 * 1024


def compile_template(text):
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                 extensions=['jinja2.ext.loopcontrols'])
    # Preserve tokenizer JSON semantics, including insertion order and Unicode.
    environment.filters['tojson'] = partial(json.dumps, ensure_ascii=False, allow_nan=False)
    def raise_exception(message):
        raise ValueError(message)
    environment.globals.update(raise_exception=raise_exception,
                               strftime_now=lambda fmt: datetime.now().strftime(fmt))
    return environment.from_string(text)


def validate_override(text, config=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('chat_template must be nonempty template text')
    try:
        size = len(text.encode('utf-8'))
    except UnicodeError as error:
        raise ValueError('chat_template must be valid UTF-8') from error
    if size > MAX_TEMPLATE_BYTES or '\x00' in text:
        raise ValueError('chat_template must be at most 128 KiB and contain no NUL')
    try:
        compile_template(text)
    except (TemplateError, RecursionError) as error:
        raise ValueError('invalid chat_template syntax') from error
    if config is not None and (config.get('task') in ('embedding', 'rerank') or
            config.get('model_type') in ('mamba', 'mamba2', 't5')):
        raise ValueError('chat_template overrides require a chat-capable decoder')
    return text


def read_override(path):
    with Path(path).expanduser().open('rb') as file:
        data = file.read(MAX_TEMPLATE_BYTES + 1)
    if len(data) > MAX_TEMPLATE_BYTES:
        raise ValueError('chat_template must be at most 128 KiB')
    return validate_override(data.decode('utf-8'))


def override_metadata(text):
    return {'source': 'checkpoint' if text is None else 'override',
            'sha256': None if text is None else hashlib.sha256(text.encode('utf-8')).hexdigest()}
