"""Internal request context; never selected by HTTP payloads or bearer text.

HTTP handlers assign opaque scope identities. Pending captures the current scope
before queue handoff because worker threads do not inherit request context.
"""
from contextvars import ContextVar

cache_scope = ContextVar('dllm_cache_scope', default=None)
