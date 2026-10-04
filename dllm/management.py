"""Validation for dllm's local model-management extensions."""


def quantization_mode(value=None):
    if value is None:
        import os
        value = os.environ.get('DLLM_QUANTIZATION','none')
    if value not in ('none','q8_0','q4_0'):
        raise ValueError('quantization must be none, q8_0 or q4_0')
    return value


def command(action, model=None, path=None, release_host=False, quantization=None, chat_template=None):
    if action not in ('status', 'register', 'load', 'unload'):
        raise ValueError('unknown model management action')
    if type(release_host) is not bool:
        raise ValueError('release_host must be boolean')
    if action == 'status':
        if model is not None or path is not None or release_host:
            raise ValueError('status does not accept model options')
    elif not isinstance(model, str) or not 1 <= len(model) <= 128 or any(ord(c) < 32 for c in model):
        raise ValueError('model must be a nonempty alias of at most 128 characters')
    if action == 'register':
        if not isinstance(path, str) or not path or len(path) > 4096 or '\x00' in path:
            raise ValueError('register requires a local snapshot path')
    elif path is not None:
        raise ValueError('path is only supported by register')
    if release_host and action != 'unload':
        raise ValueError('release_host is only supported by unload')
    result = {'action': action, 'model': model, 'path': path, 'release_host': release_host}
    if quantization is not None:
        if action != 'register':
            raise ValueError('quantization is only supported by register')
        result['quantization'] = quantization_mode(quantization)
    if chat_template is not None:
        if action != 'register':
            raise ValueError('chat_template is only supported by register')
        from .chat_template import validate_override
        result['chat_template'] = validate_override(chat_template)
    return result
