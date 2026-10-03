"""Model residency and serialized request lifetime for the first GPU executor."""

from collections import OrderedDict
from contextlib import contextmanager
import gc
import os
import threading
import time


def require_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one GPU UUID")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")
    torch.cuda.set_device(0)


class BusyError(Exception):
    pass


class Runtime:
    def __init__(self, models, context=1024, gpu_models=1, host_models=2,
                 host_gib=8, use_graph=True, queue_size=8, queue_timeout=120):
        if (not models or min(context, gpu_models, host_models, queue_size) <= 0
                or host_models < gpu_models or host_gib <= 0 or queue_timeout <= 0):
            raise ValueError("invalid model, residency, context, or queue limits")
        require_gpu()
        self.models = dict(models)
        self.context = context
        self.gpu_limit, self.host_limit = gpu_models, host_models
        self.host_bytes = int(host_gib * 1024**3)
        self.use_graph = use_graph
        self.queue_timeout = queue_timeout
        self.host = OrderedDict()
        self.gpu = OrderedDict()
        self.lock = threading.Lock()
        self.admission = threading.BoundedSemaphore(queue_size)

    def _evict(self, alias):
        import torch
        torch.cuda.synchronize()
        del self.gpu[alias]
        gc.collect()
        torch.cuda.empty_cache()

    def _activate(self, alias):
        import torch
        from .qwen2 import ModelWeights, Executor, weight_bytes
        start = time.perf_counter()
        if alias in self.gpu:
            self.gpu.move_to_end(alias)
            self.host.move_to_end(alias)
            return self.gpu[alias], "gpu", time.perf_counter() - start
        tier = "host" if alias in self.host else "disk"
        if alias not in self.host:
            estimated = weight_bytes(self.models[alias])
            if estimated > self.host_bytes:
                raise MemoryError("model exceeds configured host weight budget")
            # Evict host entries only when their GPU executors are also idle;
            # this method always runs under the request-lifetime lock.
            while (len(self.host) >= self.host_limit or
                   sum(m.bytes for m in self.host.values()) + estimated > self.host_bytes):
                victim = next(iter(self.host))
                if victim in self.gpu:
                    self._evict(victim)
                del self.host[victim]
            model = ModelWeights(self.models[alias], self.context)
            if model.bytes > self.host_bytes:
                raise MemoryError("model exceeds configured host weight budget")
            while sum(m.bytes for m in self.host.values()) + model.bytes > self.host_bytes:
                victim = next(iter(self.host))
                if victim in self.gpu:
                    self._evict(victim)
                del self.host[victim]
            self.host[alias] = model
        self.host.move_to_end(alias)
        while len(self.gpu) >= self.gpu_limit:
            self._evict(next(iter(self.gpu)))
        # Failure never publishes a partially initialized executor as ready.
        try:
            executor = Executor(self.host[alias], self.use_graph)
        except Exception:
            gc.collect()
            torch.cuda.empty_cache()
            raise
        self.gpu[alias] = executor
        torch.cuda.synchronize()
        return executor, tier, time.perf_counter() - start

    @contextmanager
    def session(self, alias):
        if alias not in self.models:
            raise ValueError("unknown model")
        if not self.admission.acquire(blocking=False):
            raise BusyError("request queue is full")
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=self.queue_timeout)
            if not acquired:
                raise BusyError("request queue timeout")
            yield self._activate(alias)
        finally:
            if acquired:
                self.lock.release()
            self.admission.release()

    def generate(self, alias, messages, max_tokens=128, ignore_eos=False):
        """Yield token IDs/text deltas plus a final usage/timing event.

        Closing this generator releases its model lease, including on client
        disconnect. A single lease spans loading, prefill, and all decoding.
        """
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        start = time.perf_counter()
        with self.session(alias) as (executor, tier, load_seconds):
            model = executor.model
            prompt = model.chat_tokens(messages)
            if len(prompt) + max_tokens > self.context:
                raise ValueError("prompt plus max_tokens exceeds configured context")
            first = executor.prefill(prompt).argmax(-1).item()
            output = []
            emitted = ""
            first_time = None
            token = first
            finish = "length"
            for index in range(max_tokens):
                output.append(token)
                if first_time is None:
                    first_time = time.perf_counter() - start
                stop = not ignore_eos and token in model.eos
                decoded = model.tokenizer.decode(output, skip_special_tokens=True)
                # Wait for complete UTF-8 sequences. Tokenizer decode can end
                # with a replacement character for a partial byte sequence.
                if not decoded.endswith("\ufffd") or stop or index == max_tokens - 1:
                    if not decoded.startswith(emitted):
                        raise RuntimeError("tokenizer revised already streamed text")
                    delta = decoded[len(emitted):]
                    emitted = decoded
                    yield {"token": token, "text": delta}
                else:
                    yield {"token": token, "text": ""}
                if stop:
                    finish = "stop"
                    break
                if index + 1 < max_tokens:
                    _, next_token = executor.decode(token)
                    token = next_token.item()
            elapsed = time.perf_counter() - start
            yield {"finish_reason": finish, "usage": {
                "prompt_tokens": len(prompt), "completion_tokens": len(output),
                "total_tokens": len(prompt) + len(output),
            }, "timing": {"source_tier": tier, "activation_seconds": load_seconds,
                          "first_token_seconds": first_time, "total_seconds": elapsed,
                          "decode_tokens_per_second": (len(output) - 1) / (elapsed - first_time)
                          if len(output) > 1 and elapsed > first_time else None}}
