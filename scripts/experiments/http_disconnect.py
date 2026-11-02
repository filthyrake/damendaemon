"""Deterministic urllib disconnects for HTTP lifecycle harnesses (#664).

urllib's response.close() can leave the socket in a reference cycle — the
FIN then waits for cyclic GC, which on a quiet harness can be indefinitely
late. Measured on the Q4 32K cancel cell: the "zombie" request was the
harness socket never closing, so the server could not observe the
disconnect at any level (no FIN → no write error → no cancellation); the
GPU worker decoded the entire request for a client that was already gone.
"""


def hard_disconnect(response):
    """FIN a urllib response connection deterministically: chase the socket
    through the response's file chain and close it explicitly, then force a
    cyclic collect as the fallback when the chain's shape changes."""
    import gc
    socket_obj = response
    for attribute in ('fp', 'fp', 'raw', '_sock'):
        socket_obj = getattr(socket_obj, attribute, None)
        if socket_obj is None:
            break
    if socket_obj is not None:
        try:
            socket_obj.close()
        except OSError:
            pass
    gc.collect()
