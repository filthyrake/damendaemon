"""Exception snapshots for long-lived worker status and cross-thread results."""
import copy


def detached_error(error):
    """Retain error semantics without retaining worker execution frames."""
    try:
        if isinstance(error,BaseExceptionGroup):
            detached=error.derive([detached_error(child) for child in error.exceptions])
        else:
            detached=copy.copy(error)
        detached=detached.with_traceback(None)
    except Exception:
        detached=RuntimeError(str(error))
    detached.__cause__=detached.__context__=None
    detached.__notes__=list(getattr(error,'__notes__',()))
    return detached

