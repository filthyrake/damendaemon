"""CPU-boundary device stages for the authenticated host worker protocol.

Wraps a fresh eager ``HybridLayerStage`` whose weights and state live on an
accelerator (CUDA, Intel XPU or Apple MPS) behind the exact CPU surface
``HostHybridWorker`` drives: token IDs and boundary activations arrive as
ordinary CPU tensors, execution happens on the stage's device, and results
return as blocking contiguous CPU copies. The wire contract stays the
existing CPU FP32 one: FP16 compute widens exactly to the FP32 boundary, and
the inbound narrowing back to FP16 is bit-preserving for the finite values
the protocol validates.

Compute precision follows each device's qualified policy and may legitimately
differ between the two ranks of a pair; the worker exchanges and records the
per-rank descriptors at handshake. Transfers are blocking, unpinned and
bounded per call; bytes and copy/compute seconds are counted for the cost
decomposition. No streams, peer access, direct device transport or serving
admission is implied. This is the device-worker building block for
cross-vendor execution, not a qualified performance claim.
"""
import threading
import time
from contextlib import nullcontext

import torch

from .hybrid_stage import HybridLayerStage


# The two ranks of an in-process device pair close concurrently when a session
# ends or fails. Metal command buffers cannot interleave synchronize and
# cache-release commits from two threads, so device-side close/synchronize is
# serialized per process; execution itself stays interleaved by the protocol.
_DEVICE_CLOSE_LOCK = threading.Lock()


def _budget(required, limit):
    if type(limit) is not int or not 1 <= limit <= 2**31 or required > limit:
        raise ValueError('device boundary transfer payload budget exceeded')


class DeviceHybridStage:
    """Own a fresh device stage behind the CPU worker boundary contract."""

    boundary_dtype = torch.float32

    def __init__(self, stage, *, max_transfer_bytes):
        if not isinstance(stage, HybridLayerStage):
            raise ValueError('device boundary requires a hybrid layer stage')
        if (stage.closed or getattr(stage, 'failed', False)
                or getattr(stage, '_pipeline_owner', None) is not None
                or any(stage.lengths)):
            raise ValueError('device boundary requires a fresh unowned stage')
        if stage.actual_device.type == 'cpu':
            raise ValueError('device boundary requires a non-CPU stage device')
        if stage.compute_dtype not in (torch.float32, torch.float16):
            raise ValueError('device boundary requires FP32 or FP16 compute')
        _budget(0, max_transfer_bytes)
        self.stage = stage
        self.max_transfer_bytes = max_transfer_bytes
        self.closed = False
        self.stats = dict(calls=0, bytes_in=0, bytes_out=0,
                         copy_seconds=0.0, compute_seconds=0.0)

    # -- ownership surface passed through to the wrapped stage -----------

    @property
    def c(self):
        return self.stage.c

    @property
    def context(self):
        return self.stage.context

    @property
    def max_batch(self):
        return self.stage.max_batch

    @property
    def first(self):
        return self.stage.first

    @property
    def last(self):
        return self.stage.last

    @property
    def layer_start(self):
        return self.stage.layer_start

    @property
    def layer_end(self):
        return self.stage.layer_end

    @property
    def kv(self):
        return self.stage.kv

    @property
    def conv(self):
        return self.stage.conv

    @property
    def state(self):
        return self.stage.state

    @property
    def w(self):
        return self.stage.w

    @property
    def model(self):
        return self.stage.model

    @property
    def actual_device(self):
        return self.stage.actual_device

    @property
    def compute_dtype(self):
        return self.stage.compute_dtype

    @property
    def lengths(self):
        return self.stage.lengths

    @property
    def length(self):
        return self.stage.length

    @length.setter
    def length(self, value):
        self.stage.length = value

    def synchronize(self):
        if not self.closed:
            with _DEVICE_CLOSE_LOCK:
                self.stage.synchronize()

    def close(self):
        if self.closed:
            return
        with _DEVICE_CLOSE_LOCK:
            self.stage.close()
        self.closed = True

    # -- boundary helpers -------------------------------------------------

    def _device_context(self):
        device = self.stage.actual_device
        if device.type == 'cuda':
            return torch.cuda.device(device)
        return nullcontext()

    def _to_device(self, value):
        # The hybrid executor's residual stream is FP32 by design (embedding
        # returns .float() and the residual accumulates in FP32); compute
        # precision lives in the weights and attention internals. Boundary
        # tensors therefore move to the device unchanged.
        return value.to(device=self.stage.actual_device, copy=True).contiguous()

    def _to_host(self, value):
        return value.to(device='cpu', dtype=self.boundary_dtype,
                        copy=True).contiguous()

    def _record(self, payload_in, payload_out, started, compute):
        elapsed = time.perf_counter() - started
        stats = self.stats
        stats['calls'] += 1
        stats['bytes_in'] += payload_in
        stats['bytes_out'] += payload_out
        if compute:
            stats['compute_seconds'] += elapsed
        else:
            stats['copy_seconds'] += elapsed

    def _host_tokens(self, tokens, batch=None, length=None):
        if (not isinstance(tokens, torch.Tensor) or tokens.device.type != 'cpu'
                or tokens.dtype != torch.int64 or tokens.ndim != 2
                or any(expected is not None and actual != expected
                       for actual, expected in zip(tokens.shape, (batch, length)))
                or min(tokens.shape) < 1):
            raise ValueError('device boundary requires matching CPU int64 token geometry')
        if tokens.shape[0] > self.stage.max_batch or tokens.shape[1] > self.stage.context:
            raise ValueError('device boundary token extent exceeds slots/context')
        return tokens.numel() * 8

    def _host_hidden(self, hidden, *, output=False):
        # forward_layers consumes (batch, rows, width); the eager head consumes
        # the final row per slot: (batch, width).
        expected = 2 if output else 3
        if (not isinstance(hidden, torch.Tensor) or hidden.device.type != 'cpu'
                or hidden.dtype != torch.float32 or hidden.ndim != expected
                or not (output or hidden.shape[1] <= self.stage.context)
                or hidden.shape[0] > self.stage.max_batch):
            raise ValueError('device boundary requires matching CPU FP32 activation geometry')
        return hidden.numel() * 4

    # -- worker-facing execution surface ----------------------------------

    @torch.no_grad()
    def embedding(self, tokens):
        if self.closed:
            raise RuntimeError('device boundary stage is closed')
        size = self._host_tokens(tokens)
        width = self.stage.c['hidden_size']
        output = tokens.numel() * width * 4
        _budget(size + output, self.max_transfer_bytes)
        started = time.perf_counter()
        with self._device_context():
            result = self._to_host(self.stage.embedding(self._to_device(tokens)))
        self._record(size, output, started, compute=True)
        return result

    @torch.no_grad()
    def forward_layers(self, hidden, positions, slots, prefill, prefill_start=0):
        if self.closed:
            raise RuntimeError('device boundary stage is closed')
        size = self._host_hidden(hidden)
        if (not isinstance(positions, torch.Tensor) or positions.device.type != 'cpu'
                or positions.dtype != torch.int64 or positions.shape != hidden.shape[:2]
                or not isinstance(slots, torch.Tensor) or slots.device.type != 'cpu'
                or slots.dtype != torch.int64 or slots.shape != (hidden.shape[0],)):
            raise ValueError('device boundary positions/slots must match the activation batch')
        size += positions.numel() * 8 + slots.numel() * 8
        output = hidden.numel() * 4
        _budget(size + output, self.max_transfer_bytes)
        started = time.perf_counter()
        with self._device_context():
            result = self._to_host(self.stage.forward_layers(
                self._to_device(hidden),
                self._to_device(positions), self._to_device(slots),
                prefill, prefill_start))
        self._record(size, output, started, compute=True)
        return result

    @torch.no_grad()
    def output_logits(self, hidden):
        if self.closed:
            raise RuntimeError('device boundary stage is closed')
        if (not self.stage.last):
            raise ValueError('only the final stage projects logits')
        size = self._host_hidden(hidden, output=True)
        output = hidden.shape[0] * self.stage.c['vocab_size'] * 4
        _budget(size + output, self.max_transfer_bytes)
        started = time.perf_counter()
        with self._device_context():
            result = self._to_host(self.stage.output_logits(self._to_device(hidden)))
        self._record(size, output, started, compute=True)
        return result
