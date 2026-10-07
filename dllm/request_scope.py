"""Internal request context; never selected by HTTP payloads or bearer text.

HTTP handlers assign opaque scope identities. Pending captures the current scope
before queue handoff because worker threads do not inherit request context.
"""
from contextvars import ContextVar
from dataclasses import dataclass

cache_scope = ContextVar('dllm_cache_scope', default=None)


@dataclass(frozen=True, eq=False)
class CacheScope:
    """Identity-based ownership with immutable optional per-cache capacity."""
    max_prefix_cache_tokens: int | None = None

    def __post_init__(self):
        limit = self.max_prefix_cache_tokens
        if limit is not None and (type(limit) is not int or not 0 <= limit <= 2**31-1):
            raise ValueError('invalid scope prefix cache token limit')
