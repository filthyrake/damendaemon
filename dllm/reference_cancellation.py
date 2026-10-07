"""Cooperative CPU token-boundary cancellation, scoped to one execution context."""
from contextlib import contextmanager
from contextvars import ContextVar

_callbacks=ContextVar('dllm_reference_cancellation',default=())


class ReferenceCancelled(RuntimeError):
    pass


@contextmanager
def cancellation_scope(cancelled):
    if not callable(cancelled):raise ValueError('reference cancellation requires a callable')
    token=_callbacks.set(_callbacks.get()+(cancelled,))
    try:yield
    finally:_callbacks.reset(token)


def check_cancelled():
    if any(cancelled() for cancelled in _callbacks.get()):
        raise ReferenceCancelled('CPU reference generation cancelled')
