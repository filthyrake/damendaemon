"""Per-client outstanding HTTP inference limits, independent of queue capacity."""
from contextlib import contextmanager
import threading
import time
import math
from collections import deque


class ClientLimitError(Exception):
    pass


class ClientOutputLimitError(ClientLimitError):
    pass


class ClientRateLimitError(ClientLimitError):
    def __init__(self,retry_after):
        super().__init__('client inference request rate limit reached')
        self.retry_after=retry_after


class ClientAdmission:
    def __init__(self, auth, *, clock=None):
        self.auth = auth
        self.active = {}
        self.requests = {}
        self.clock = time.monotonic if clock is None else clock
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
            if client.max_requests_per_minute is not None:
                now=self.clock()
                history=self.requests.setdefault(client.name,deque())
                while history and history[0]<=now-60:history.popleft()
                if len(history)>=client.max_requests_per_minute:
                    raise ClientRateLimitError(max(1,math.ceil(history[0]+60-now)))
                # Charge accepted inference admission attempts, even if the
                # request later fails validation, disconnects or cancels.
                history.append(now)
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
