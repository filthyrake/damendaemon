"""Own local verifier HTTP startup and cleanup, including partial startup."""
from contextlib import contextmanager
import threading
from dllm.server import BoundedHTTPServer, handler_for


@contextmanager
def owned_http_server(runtime, *, server_factory=BoundedHTTPServer):
    server = thread = None
    started = False
    primary = None
    try:
        server = server_factory(('127.0.0.1', 0), handler_for(runtime))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started = True
        yield server
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        def cleanup(action):
            try: action()
            except BaseException as error: errors.append(error)
        # shutdown() waits forever if serve_forever() never started.
        if started: cleanup(server.shutdown)
        cleanup(runtime.close)
        if server is not None: cleanup(server.server_close)
        if started:
            cleanup(lambda: thread.join(timeout=5))
            if thread.is_alive(): errors.append(RuntimeError('verifier HTTP thread did not stop'))
        if errors:
            if primary is not None:
                primary.add_note('Verifier cleanup also failed: '+', '.join(type(error).__name__ for error in errors))
            else:
                raise BaseExceptionGroup('verifier cleanup failed', errors)
