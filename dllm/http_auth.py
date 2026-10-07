"""Optional loopback API roles and redirect-safe authenticated clients."""
from dataclasses import dataclass, field
import json
import os
import re
import secrets
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, build_opener, urlopen

INFERENCE_ROUTES = frozenset(('/v1/chat/completions', '/v1/completions', '/v1/embeddings',
                              '/v1/token-scores', '/v1/rerank'))


def _key(name, value):
    if value is None:
        return None
    if (not isinstance(value, str) or not 16 <= len(value) <= 512
            or not value.isascii() or any(not 33 <= ord(c) <= 126 for c in value)):
        raise ValueError(f'{name} must be 16..512 printable ASCII characters without spaces')
    return value


@dataclass(frozen=True)
class ClientKey:
    name: str
    key: str = field(repr=False)
    max_inflight: int
    models: frozenset | None = None
    max_reserved_output_tokens: int | None = None
    max_prefix_cache_tokens: int | None = None
    max_requests_per_minute: int | None = None
    max_output_tokens_per_minute: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', self.name) is None:
            raise ValueError('client names must contain 1..64 letters, digits, dots, underscores or hyphens')
        _key('client key', self.key)
        if self.key is None:
            raise ValueError('client key is required')
        if type(self.max_inflight) is not int or not 1 <= self.max_inflight <= 128:
            raise ValueError('client max_inflight must be an integer from 1 to 128')
        if self.max_reserved_output_tokens is not None and (type(self.max_reserved_output_tokens) is not int
                or not 1 <= self.max_reserved_output_tokens <= 2**31-1):
            raise ValueError('client max_reserved_output_tokens must be an integer from 1 to 2147483647')
        if self.max_prefix_cache_tokens is not None and (type(self.max_prefix_cache_tokens) is not int
                or not 0 <= self.max_prefix_cache_tokens <= 2**31-1):
            raise ValueError('client max_prefix_cache_tokens must be an integer from 0 to 2147483647')
        if self.max_requests_per_minute is not None and (type(self.max_requests_per_minute) is not int
                or not 1 <= self.max_requests_per_minute <= 60000):
            raise ValueError('client max_requests_per_minute must be an integer from 1 to 60000')
        if self.max_output_tokens_per_minute is not None and (type(self.max_output_tokens_per_minute) is not int
                or not 1 <= self.max_output_tokens_per_minute <= 2**31-1):
            raise ValueError('client max_output_tokens_per_minute must be an integer from 1 to 2147483647')
        if self.models is not None:
            if not isinstance(self.models, frozenset) or len(self.models) > 128:
                raise ValueError('client models must be a frozenset of at most 128 exact aliases')
            from .management import command
            for alias in self.models:
                command('load', model=alias)

    def allows_model(self, alias):
        return self.models is None or alias in self.models


@dataclass(frozen=True)
class AuthPolicy:
    inference_key: str | None = field(default=None, repr=False)
    admin_key: str | None = field(default=None, repr=False)
    clients: tuple = field(default=(), repr=False)

    def __post_init__(self):
        _key('DLLM_API_KEY', self.inference_key)
        _key('DLLM_ADMIN_API_KEY', self.admin_key)
        if self.inference_key is not None and self.admin_key is not None and secrets.compare_digest(self.inference_key, self.admin_key):
            raise ValueError('inference and administrative keys must differ')
        if not isinstance(self.clients, tuple) or len(self.clients) > 128 or any(not isinstance(c, ClientKey) for c in self.clients):
            raise ValueError('configure at most 128 named clients')
        keys = [value for value in (self.inference_key, self.admin_key) if value is not None]
        keys.extend(c.key for c in self.clients)
        if len(set(keys)) != len(keys) or len({c.name for c in self.clients}) != len(self.clients):
            raise ValueError('client names and all API keys must be distinct')

    @classmethod
    def from_env(cls):
        clients = ()
        text = os.environ.get('DLLM_CLIENT_KEYS')
        if text is not None:
            try:
                def unique_pairs(pairs):
                    result = {}
                    for name, value in pairs:
                        if name in result:
                            raise ValueError('duplicate field')
                        result[name] = value
                    return result
                raw = json.loads(text, object_pairs_hook=unique_pairs)
                if (not isinstance(raw, dict) or not 1 <= len(raw) <= 128
                        or any(not isinstance(value, dict) or not {'key', 'max_inflight'} <= set(value)
                               or set(value) - {'key', 'max_inflight', 'models', 'max_reserved_output_tokens', 'max_prefix_cache_tokens', 'max_requests_per_minute', 'max_output_tokens_per_minute'}
                               for value in raw.values())):
                    raise ValueError('invalid client configuration')
                parsed = []
                for name, value in raw.items():
                    models = None
                    if 'models' in value:
                        aliases = value['models']
                        if (not isinstance(aliases, list) or len(aliases) > 128
                                or any(not isinstance(alias, str) for alias in aliases)
                                or len(set(aliases)) != len(aliases)):
                            raise ValueError('invalid client model aliases')
                        models = frozenset(aliases)
                    if 'max_reserved_output_tokens' in value and value['max_reserved_output_tokens'] is None:
                        raise ValueError('explicit output reservation limit cannot be null')
                    if 'max_prefix_cache_tokens' in value and value['max_prefix_cache_tokens'] is None:
                        raise ValueError('explicit prefix cache limit cannot be null')
                    if 'max_requests_per_minute' in value and value['max_requests_per_minute'] is None:
                        raise ValueError('explicit request rate limit cannot be null')
                    if 'max_output_tokens_per_minute' in value and value['max_output_tokens_per_minute'] is None:
                        raise ValueError('explicit output rate limit cannot be null')
                    parsed.append(ClientKey(name, value['key'], value['max_inflight'], models,
                                            value.get('max_reserved_output_tokens'), value.get('max_prefix_cache_tokens'),
                                            value.get('max_requests_per_minute'), value.get('max_output_tokens_per_minute')))
                clients = tuple(parsed)
            except (ValueError, TypeError):
                raise ValueError('DLLM_CLIENT_KEYS requires valid distinct clients with key/max_inflight and optional models/max_reserved_output_tokens/max_prefix_cache_tokens/max_requests_per_minute/max_output_tokens_per_minute') from None
        return cls(os.environ.get('DLLM_API_KEY'), os.environ.get('DLLM_ADMIN_API_KEY'), clients)

    def client(self, headers):
        if len(headers) != 1:
            return None
        scheme, separator, token = headers[0].partition(' ')
        if not separator or scheme.lower() != 'bearer' or not token.isascii() or not 16 <= len(token) <= 512:
            return None
        matches = [client for client in self.clients if secrets.compare_digest(token, client.key)]
        return matches[0] if matches else None

    def authorize(self, method, path, headers):
        if self.inference_key is None and self.admin_key is None and not self.clients:
            return None
        if method == 'GET' and path == '/health':
            return None
        if len(headers) != 1:
            return 401
        scheme, separator, token = headers[0].partition(' ')
        if (not separator or scheme.lower() != 'bearer' or not token.isascii()
                or not 16 <= len(token) <= 512):
            return 401
        admin = self.admin_key is not None and secrets.compare_digest(token, self.admin_key)
        inference = ((self.inference_key is not None and secrets.compare_digest(token, self.inference_key))
                     or self.client(headers) is not None)
        if not admin and not inference:
            return 401
        if admin or (method == 'GET' and path == '/v1/models') or (method == 'POST' and path in INFERENCE_ROUTES):
            return None
        return 403


def management_headers():
    policy = AuthPolicy.from_env()
    key = policy.admin_key or policy.inference_key
    return {} if key is None else {'Authorization': 'Bearer '+key}


class _NoAuthenticatedRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, 'authenticated redirects are disabled', headers, fp)


def authenticated_urlopen(request, **kwargs):
    if hasattr(request, 'get_header') and request.get_header('Authorization') is not None:
        return build_opener(_NoAuthenticatedRedirect()).open(request, **kwargs)
    return urlopen(request, **kwargs)
