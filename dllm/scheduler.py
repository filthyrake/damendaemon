"""Continuous decode batching with bounded admission and cancellable streams.

One worker owns the GPU. Requests join/free KV slots between decode steps.
Model groups drain before switching; a queued different model prevents new
admission to the current group, avoiding starvation and active-model eviction.
"""
from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time

from .runtime import Runtime, BusyError
from .sampling import Sampling, Sampler, sample_rows
from .stopping import stop_sequences, text_delta


@dataclass(eq=False)
class Pending:
    alias: str
    messages: list
    max_tokens: int
    enable_thinking: bool | None = None
    sampler: Sampler = field(default_factory=lambda: Sampler(Sampling()))
    stops: tuple = ()
    start: float = field(default_factory=time.perf_counter)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8))
    done: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    final: dict | None = None
    output: list = field(default_factory=list)
    emitted: str = ""
    first_time: float | None = None
    last_progress: float = field(default_factory=time.monotonic)
    max_batch_seen: int = 1
    slot: int = -1
    tier: str = "gpu"
    activation: float = 0


class BatchedRuntime:
    def __init__(self, models, batch_size=4, queue_size=8, queue_timeout=120, **kwargs):
        if not 1 <= batch_size <= 16 or queue_size < batch_size or queue_timeout <= 0:
            raise ValueError("batch size must be 1..16 and fit inside the request queue")
        self.runtime = Runtime(models, batch_size=batch_size, **kwargs)
        self.models = self.runtime.models
        self.batch_size = batch_size
        self.queue_timeout = queue_timeout
        self.admission = threading.BoundedSemaphore(queue_size)
        self.condition = threading.Condition()
        self.pending = deque()
        self.requests = set()
        self.closed = False
        self.worker = threading.Thread(target=self._work, name="dllm-gpu", daemon=True)
        self.worker.start()

    def generate(self, alias, messages, max_tokens=128, enable_thinking=None,
                 temperature=0, top_p=1, top_k=0, seed=None, stop=None):
        if alias not in self.models:
            raise ValueError("unknown model")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        sampler = Sampler(Sampling(temperature, top_p, top_k, seed))
        stops = stop_sequences(stop)
        if not self.admission.acquire(blocking=False):
            raise BusyError("request queue is full")
        request = Pending(alias, messages, max_tokens, enable_thinking, sampler, stops)
        try:
            with self.condition:
                if self.closed:
                    raise BusyError("runtime is closed")
                self.requests.add(request)
                self.pending.append(request)
                self.condition.notify_all()
            while True:
                request.ready.clear()
                try:
                    event = request.events.get_nowait()
                except queue.Empty:
                    if request.done.is_set():
                        if request.error:
                            raise request.error
                        if request.final:
                            yield request.final
                        return
                    request.ready.wait()
                    continue
                with self.condition:
                    self.condition.notify_all()
                yield event
        finally:
            request.cancelled.set()
            with self.condition:
                try:
                    self.pending.remove(request)
                except ValueError:
                    pass
                self.requests.discard(request)
                self.condition.notify_all()
            self.admission.release()

    def _finish(self, request, error=None):
        with self.condition:
            if request.done.is_set():
                return
            request.error = error
            request.done.set()
            request.ready.set()
            self.requests.discard(request)

    def _emit(self, request, token, executor, prompt_count):
        if request.cancelled.is_set():
            self._finish(request)
            return True
        request.output.append(token)
        now = time.perf_counter()
        if request.first_time is None:
            request.first_time = now - request.start
        request.last_progress = time.monotonic()
        stop = token in executor.model.eos
        finished = stop or len(request.output) >= request.max_tokens
        decoded = executor.model.tokenizer.decode(request.output, skip_special_tokens=True)
        delta, request.emitted, text_stop = text_delta(decoded, request.emitted, request.stops, finished)
        stop = stop or text_stop
        finished = finished or text_stop
        request.events.put_nowait({"token": token, "text": delta})
        request.ready.set()
        if finished:
            elapsed = time.perf_counter() - request.start
            count = len(request.output)
            request.final = {"finish_reason": "stop" if stop else "length", "usage": {
                "prompt_tokens": prompt_count, "completion_tokens": count,
                "total_tokens": prompt_count + count}, "timing": {
                "source_tier": request.tier, "activation_seconds": request.activation,
                "kernel_backend": executor.kernel_backend, "max_batch_size": request.max_batch_seen,
                "residual_dtype": str(executor.residual_dtype).removeprefix("torch."),
                "first_token_seconds": request.first_time, "total_seconds": elapsed,
                "decode_tokens_per_second": (count-1)/(elapsed-request.first_time)
                if count > 1 and elapsed > request.first_time else None}}
            self._finish(request)
        return finished

    def _next(self, alias=None):
        """Caller holds the condition. Preserve FIFO across model groups."""
        while self.pending:
            first = self.pending[0]
            if first.cancelled.is_set():
                self.pending.popleft()
                self._finish(first)
            elif time.perf_counter() - first.start > self.queue_timeout:
                self.pending.popleft()
                self._finish(first, BusyError("request queue timeout"))
            elif alias is not None and first.alias != alias:
                return None
            else:
                return self.pending.popleft()
        return None

    def _group(self, first, executor, tier, activation):
        first.tier, first.activation = tier, activation
        active = {}  # slot -> (request, prompt length)
        next_request = first
        try:
            while next_request is not None or active:
                if next_request is not None:
                    incoming = [next_request]
                    next_request = None
                    with self.condition:
                        while len(incoming) + len(active) < self.batch_size:
                            request = self._next(first.alias)
                            if request is None:
                                break
                            incoming.append(request)
                    prepared = []
                    free_slots = iter(i for i in range(self.batch_size) if i not in active)
                    for request in incoming:
                        try:
                            if request.cancelled.is_set():
                                self._finish(request)
                                continue
                            tokens = executor.model.chat_tokens(request.messages, **(
                                {"enable_thinking": request.enable_thinking} if request.enable_thinking is not None else {}))
                            if len(tokens) + request.max_tokens > executor.context:
                                raise ValueError("prompt plus max_tokens exceeds configured context")
                            request.slot = next(free_slots)
                            prepared.append((request, tokens))
                        except Exception as error:
                            self._finish(request, error)
                    if prepared:
                        try:
                            logits = executor.prefill_batch(
                                [tokens for _, tokens in prepared],
                                [request.slot for request, _ in prepared])
                            first_tokens = sample_rows(logits, [r.sampler for r, _ in prepared])
                            for (request, tokens), token in zip(prepared, first_tokens):
                                if not self._emit(request, token, executor, len(tokens)):
                                    active[request.slot] = (request, len(tokens))
                        except Exception as error:
                            for request, _ in prepared:
                                self._finish(request, error)
                                active.pop(request.slot, None)
                for slot, (request, _) in list(active.items()):
                    if request.cancelled.is_set():
                        self._finish(request)
                        del active[slot]
                    elif request.events.full() and time.monotonic() - request.last_progress > self.queue_timeout:
                        self._finish(request, BusyError("stream backpressure timeout"))
                        del active[slot]
                ready = [(slot, request, length) for slot, (request, length) in active.items()
                         if not request.events.full()]
                if ready:
                    logits, next_tokens = executor.decode_batch([r.output[-1] for _, r, _ in ready],
                                                           [slot for slot, _, _ in ready])
                    selected = sample_rows(logits, [r.sampler for _, r, _ in ready], next_tokens)
                    for (slot, request, length), token in zip(ready, selected):
                        request.max_batch_seen = max(request.max_batch_seen, len(ready))
                        if self._emit(request, token, executor, length):
                            del active[slot]
                with self.condition:
                    if len(active) < self.batch_size:
                        next_request = self._next(first.alias)
                    if active and not ready and next_request is None:
                        self.condition.wait(timeout=0.02)
        except Exception as error:
            for request, _ in active.values():
                self._finish(request, error)
            if next_request:
                self._finish(next_request, error)
            raise

    def _work(self):
        while True:
            with self.condition:
                first = self._next()
                while first is None and not self.closed:
                    self.condition.wait()
                    first = self._next()
                if self.closed:
                    return
            try:
                with self.runtime.session(first.alias) as (executor, tier, activation):
                    self._group(first, executor, tier, activation)
            except Exception as error:
                if not first.done.is_set():
                    self._finish(first, error)

    def close(self):
        with self.condition:
            self.closed = True
            for request in tuple(self.requests):
                request.cancelled.set()
                self._finish(request, BusyError("runtime closed"))
            self.condition.notify_all()
        self.worker.join(timeout=30)
