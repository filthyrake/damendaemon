"""Explicit model identities for repeatable server startup; no runtime state."""
import json
from pathlib import Path
import re
import os
import tempfile

from .management import command, quantization_mode


def encode(data):
    content = (json.dumps(data, indent=2)+'\n').encode('utf-8')
    if len(content) > 1024 * 1024:
        raise ValueError('model manifest exceeds 1 MiB')
    return content


def snapshot(models, policies, templates):
    if not models:
        raise ValueError('no registered models to export')
    entries = {}
    for alias, path in models.items():
        path = str(Path(path).expanduser().resolve())
        command('register', alias, path, quantization=policies[alias],
                chat_template=templates[alias])
        entry = dict(path=path, quantization=policies[alias])
        if templates[alias] is not None:
            entry['chat_template'] = templates[alias]
        entries[alias] = entry
    result = dict(version=1, models=entries)
    encode(result)
    return result


def save(path, data):
    """Publish a complete export atomically; failed writes preserve old content."""
    content = encode(data)
    path = Path(path).expanduser()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate manifest key: {key}')
        result[key] = value
    return result


def load(path, resolve):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('model manifest exceeds 1 MiB')
    data = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or set(data) != {'version', 'models'}
            or type(data['version']) is not int or data['version'] != 1
            or not isinstance(data['models'], dict) or not data['models']):
        raise ValueError('model manifest requires version 1 and a nonempty models object')
    entries = []
    for alias, entry in data['models'].items():
        if not isinstance(entry, dict) or set(entry) - {
                'path', 'repository', 'revision', 'filename', 'quantization', 'chat_template'}:
            raise ValueError('unknown model manifest fields')
        if ('path' in entry) == ('repository' in entry):
            raise ValueError('each model requires exactly one path or pinned repository')
        policy = entry.get('quantization')
        if policy is None:
            raise ValueError('manifest models require explicit quantization')
        quantization_mode(policy)
        template = entry.get('chat_template')
        if 'path' in entry:
            source = entry['path']
            if not isinstance(source, str) or not source or set(entry) & {'revision', 'filename'}:
                raise ValueError('local model path cannot include repository options')
            candidate = Path(source).expanduser()
            source = str((path.parent / candidate).resolve())
        else:
            source = entry['repository']
            if not isinstance(source, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', source):
                raise ValueError('manifest repository must be owner/name')
            if not isinstance(entry.get('revision'), str) or not re.fullmatch(r'[0-9a-f]{40}', entry['revision']):
                raise ValueError('manifest repository requires an immutable 40-character revision')
            if 'filename' in entry:
                from .download import gguf_selection
                gguf_selection(entry['filename'])
        command('register', alias, source, quantization=policy, chat_template=template)
        entries.append((alias, source, entry, policy, template))
    # Validate the entire document before resolving anything over the network.
    models, policies, templates = {}, {}, {}
    for alias, source, entry, policy, template in entries:
        models[alias] = (Path(source) if 'path' in entry else
                         resolve(source, entry['revision'], entry.get('filename')))
        policies[alias] = policy
        if template is not None:
            templates[alias] = template
    return models, policies, templates
