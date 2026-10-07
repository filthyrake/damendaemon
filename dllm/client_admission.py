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


class ClientOutputRateLimitError(ClientLimitError):
    def __init__(self, retry_after):
        super().__init__('client declared output token rate limit reached')
        self.retry_after = retry_after


class ClientAdmission:
    def __init__(self, auth, *, clock=None):
        self.auth = auth
        self.active = {}
        self.requests = {}
        self.output_windows = {}
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

    def _charge_output(self, client, tokens):
        """Caller holds lock; second buckets expire conservatively after 60s."""
        limit = client.max_output_tokens_per_minute
        if limit is None:
            return
        now = self.clock()
        history = self.output_windows.setdefault(client.name, deque())
        while history and history[0][0] <= now:
            history.popleft()
        used = sum(count for _, count in history)
        if tokens > limit - used:
            for expiry, count in history:
                used -= count
                if tokens <= limit - used:
                    raise ClientOutputRateLimitError(max(1, math.ceil(expiry - now)))
        expiry = math.floor(now) + 61
        if history and history[-1][0] == expiry:
            previous = history.pop()[1]
            history.append((expiry, previous + tokens))
        else:
            history.append((expiry, tokens))

    @contextmanager
    def output_lease(self, headers, tokens):
        """Charge declared output, and reserve outstanding tokens until close."""
        if type(tokens) is not int or tokens < 1:
            raise ValueError('output reservation requires a positive integer')
        client = self.auth.client(headers)
        if client is None:
            yield
            return
        limit = client.max_output_tokens_per_minute
        if limit is not None and tokens > limit:
            raise ValueError('declared output exceeds whole client rate allowance')
        cap = client.max_reserved_output_tokens
        with self.lock:
            count = self.reserved_output.get(client.name, 0)
            if cap is not None and tokens > cap-count:
                raise ClientOutputLimitError('client outstanding output token budget exceeded')
            self._charge_output(client, tokens)
            if cap is not None:
                self.reserved_output[client.name] = count+tokens
        try:
            yield
        finally:
            if cap is not None:
                with self.lock:
                    count = self.reserved_output[client.name]-tokens
                    if count:self.reserved_output[client.name] = count
                    else:del self.reserved_output[client.name]
