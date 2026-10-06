"""Optional loopback API roles and redirect-safe authenticated clients."""
from dataclasses import dataclass, field
import os
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
class AuthPolicy:
    inference_key: str | None = field(default=None, repr=False)
    admin_key: str | None = field(default=None, repr=False)

    def __post_init__(self):
        _key('DLLM_API_KEY', self.inference_key)
        _key('DLLM_ADMIN_API_KEY', self.admin_key)
        if self.inference_key is not None and self.admin_key is not None and secrets.compare_digest(self.inference_key, self.admin_key):
            raise ValueError('inference and administrative keys must differ')

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('DLLM_API_KEY'), os.environ.get('DLLM_ADMIN_API_KEY'))

    def authorize(self, method, path, headers):
        if self.inference_key is None and self.admin_key is None:
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
        inference = self.inference_key is not None and secrets.compare_digest(token, self.inference_key)
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
