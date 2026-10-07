"""Per-client outstanding HTTP inference limits, independent of queue capacity."""
from contextlib import contextmanager
import threading


class ClientLimitError(Exception):
    pass


class ClientOutputLimitError(ClientLimitError):
    pass


class ClientAdmission:
    def __init__(self, auth):
        self.auth = auth
        self.active = {}
        self.reserved_output = {}
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

    @contextmanager
    def output_lease(self, headers, tokens):
        """Reserve declared generation tokens through generator close, atomically."""
        if type(tokens) is not int or tokens < 1:
            raise ValueError('output reservation requires a positive integer')
        client = self.auth.client(headers)
        if client is None or client.max_reserved_output_tokens is None:
            yield
            return
        with self.lock:
            count = self.reserved_output.get(client.name, 0)
            if tokens > client.max_reserved_output_tokens-count:
                raise ClientOutputLimitError('client outstanding output token budget exceeded')
            self.reserved_output[client.name] = count+tokens
        try:
            yield
        finally:
            with self.lock:
                count = self.reserved_output[client.name]-tokens
                if count:self.reserved_output[client.name] = count
                else:del self.reserved_output[client.name]
