"""Per-client outstanding HTTP inference limits, independent of queue capacity."""
from contextlib import contextmanager
import threading


class ClientLimitError(Exception):
    pass


class ClientAdmission:
    def __init__(self, auth):
        self.auth = auth
        self.active = {}
        self.lock = threading.Lock()

    @contextmanager
    def lease(self, headers):
        client = self.auth.client(headers)
        if client is None:
            # Shared inference/admin keys retain their existing admission policy.
            yield
            return
        with self.lock:
            count = self.active.get(client.name, 0)
            if count >= client.max_inflight:
                raise ClientLimitError('client outstanding inference limit reached')
            self.active[client.name] = count + 1
        try:
            yield
        finally:
            with self.lock:
                count = self.active[client.name] - 1
                if count:
                    self.active[client.name] = count
                else:
                    del self.active[client.name]
