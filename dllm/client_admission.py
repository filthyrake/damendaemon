"""Per-client outstanding HTTP inference limits, independent of queue capacity."""
from contextlib import contextmanager
import threading
import time
import math
import os
from .shared_client_rates import SharedClientRates,SharedRateRejected,SharedRateUnavailable
from collections import deque
from .shared_client_reservations import SharedClientReservations,SharedReservationRejected,SharedReservationUnavailable


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
    def __init__(self, auth, *, clock=None, rate_store=None, reservation_store=None):
        self.auth = auth
        path=os.environ.get('DLLM_CLIENT_RATE_DB')
        self.rate_store=rate_store if rate_store is not None else (SharedClientRates(path,auth.clients) if path is not None else None)
        reservation_path=os.environ.get('DLLM_CLIENT_RESERVATION_DB')
        self.reservation_store=reservation_store if reservation_store is not None else (
            SharedClientReservations(reservation_path,auth.clients) if reservation_path is not None else None)
        self.pending_releases={}
        self.closed=False
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
            if self.closed:raise SharedReservationUnavailable(self.reservation_store)
            count = self.active.get(client.name, 0)
            if count >= client.max_inflight:
                raise ClientLimitError('client outstanding inference limit reached')
            claim=self._reserve(client,requests=1)
            try:
                self._charge_request(client)
            except BaseException:
                self._release(claim);raise
            self.active[client.name] = count + 1
        try:
            yield
        finally:
            with self.lock:
                try:self._release(claim)
                finally:
                    count = self.active[client.name] - 1
                    if count:self.active[client.name] = count
                    else:del self.active[client.name]

    def _charge_request(self,client):
        if client.max_requests_per_minute is not None and self.rate_store is not None:
            try:self.rate_store.charge(client,'requests',1)
            except SharedRateRejected as error:raise ClientRateLimitError(error.retry_after) from None
        elif client.max_requests_per_minute is not None:
            now=self.clock()
            history=self.requests.setdefault(client.name,deque())
            while history and history[0]<=now-60:history.popleft()
            if len(history)>=client.max_requests_per_minute:
                raise ClientRateLimitError(max(1,math.ceil(history[0]+60-now)))
            # Charge accepted inference admission attempts, even if the
            # request later fails validation, disconnects or cancels.
            history.append(now)

    def _charge_output(self, client, tokens):
        """Caller holds lock; second buckets expire conservatively after 60s."""
        limit = client.max_output_tokens_per_minute
        if limit is None:
            return
        if self.rate_store is not None:
            try:self.rate_store.charge(client,'outputs',tokens)
            except SharedRateRejected as error:raise ClientOutputRateLimitError(error.retry_after) from None
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
            if self.closed:raise SharedReservationUnavailable(self.reservation_store)
            count = self.reserved_output.get(client.name, 0)
            if cap is not None and tokens > cap-count:
                raise ClientOutputLimitError('client outstanding output token budget exceeded')
            claim=self._reserve(client,requests=0,outputs=tokens) if cap is not None else None
            try:self._charge_output(client, tokens)
            except BaseException:
                self._release(claim);raise
            if cap is not None:
                self.reserved_output[client.name] = count+tokens
        try:
            yield
        finally:
            if cap is not None:
                with self.lock:
                    try:self._release(claim)
                    finally:
                        count = self.reserved_output[client.name]-tokens
                        if count:self.reserved_output[client.name] = count
                        else:del self.reserved_output[client.name]

    def _reserve(self,client,**options):
        if self.reservation_store is None:return None
        if self.pending_releases:raise SharedReservationUnavailable(self.reservation_store)
        try:return self.reservation_store.reserve(client,**options)
        except SharedReservationRejected as error:
            kind=ClientLimitError if error.kind=='requests' else ClientOutputLimitError
            raise kind(str(error)) from None
        except SharedReservationUnavailable as error:
            if error.claim is not None:self.pending_releases[error.claim.id]=error.claim
            raise

    def _release(self,claim):
        if claim is None:return
        self.pending_releases[claim.id]=claim
        self.reservation_store.release(claim)
        self.pending_releases.pop(claim.id,None)

    def retry_releases(self):
        """Retry only ended/failed HTTP leases, never live request claims."""
        with self.lock:
            for claim in list(self.pending_releases.values()):self._release(claim)

    def close(self):
        with self.lock:
            if self.closed:return
            if self.active or self.reserved_output:raise RuntimeError('client admission still has active leases')
            for claim in list(self.pending_releases.values()):self._release(claim)
            if self.reservation_store is not None:self.reservation_store.close()
            self.closed=True
