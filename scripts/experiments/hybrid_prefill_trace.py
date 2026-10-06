"""Opt-in, synchronized prefill diagnosis for #68; never a benchmark.

Wrap the Q4 context probe with --trace FILE followed by its ordinary arguments.
Each begin event is flushed before synchronization or GPU execution, so a
terminated run still identifies unfinished work. No tensors or token IDs are saved.
"""
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import runpy
import sys
import time
from unittest.mock import patch


class Trace:
    def __init__(self, stream, synchronize, memory):
        self.stream = stream
        self.synchronize = synchronize
        self.memory = memory
        self.sequence = 0
        self.parent = ContextVar('prefill_trace_parent', default=None)

    def emit(self, event):
        self.stream.write(json.dumps(event) + '\n')
        self.stream.flush()

    def call(self, operation, function, metadata):
        self.sequence += 1
        span = self.sequence
        parent = self.parent.get()
        self.emit(dict(event='begin', span=span, parent=parent,
                       operation=operation, **metadata))
        token = self.parent.set(span)
        phase = 'synchronize_before'
        started = None
        try:
            self.synchronize()
            before = self.memory()
            started = time.perf_counter()
            phase = 'execute'
            result = function()
            phase = 'synchronize_after'
            self.synchronize()
            elapsed = time.perf_counter() - started
            self.emit(dict(event='end', span=span, seconds=elapsed,
                           memory_before=before, memory_after=self.memory()))
            return result
        except BaseException as error:
            # Do not synchronize again after GPU failure or mask it by querying memory.
            self.emit(dict(event='error', span=span, phase=phase,
                           error_type=type(error).__name__,
                           seconds=None if started is None else time.perf_counter()-started))
            raise
        finally:
            self.parent.reset(token)


@contextmanager
def trace_prefill(trace, executor_class=None, attention_module=None):
    """Install temporary experiment hooks; decode and ordinary entrypoints are untouched."""
    if executor_class is None:
        from dllm import hybrid
        executor_class, attention_module = hybrid.HybridExecutor, hybrid
    original_forward = executor_class.forward_embeddings
    original_attention = attention_module.scaled_attention
    current = ContextVar('prefill_trace_context', default=None)

    def attention(q, k, v, **kwargs):
        metadata = current.get()
        if metadata is None:
            return original_attention(q, k, v, **kwargs)
        return trace.call('scaled_attention', lambda: original_attention(q, k, v, **kwargs),
                          metadata | {'query_shape': list(q.shape), 'key_shape': list(k.shape)})

    def forward(self, x, positions, slots, prefill, prefix_length=0, return_logits=True):
        if not prefill:
            return original_forward(self, x, positions, slots, prefill, prefix_length, return_logits)
        metadata = dict(prefix_tokens=prefix_length, chunk_tokens=x.shape[1], batch=x.shape[0])
        token = current.set(metadata)
        linear = self.linear
        names = {id(weight): name for name, weight in self.w.items()}

        def projection(inputs, weight, *args, **kwargs):
            return trace.call('projection', lambda: linear(inputs, weight, *args, **kwargs),
                              metadata | {'weight': names.get(id(weight), 'unknown'),
                                          'input_shape': list(inputs.shape)})

        try:
            with patch.object(self, 'linear', projection):
                return trace.call('prefill_chunk', lambda: original_forward(
                    self, x, positions, slots, prefill, prefix_length, return_logits), metadata)
        finally:
            current.reset(token)

    def wrap_stage(original, name):
        def stage(self, x, i, *args, **kwargs):
            metadata = current.get()
            if metadata is None:
                return original(self, x, i, *args, **kwargs)
            return trace.call(name, lambda: original(self, x, i, *args, **kwargs),
                              metadata | {'layer': i})
        return stage

    with patch.object(executor_class, 'forward_embeddings', forward), \
            patch.object(executor_class, 'attention', wrap_stage(executor_class.attention, 'full_attention')), \
            patch.object(executor_class, 'linear_attention', wrap_stage(executor_class.linear_attention, 'linear_attention')), \
            patch.object(attention_module, 'scaled_attention', attention):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('probe_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.probe_args or args.probe_args[0] not in ('serve', 'replay'):
        parser.error('provide serve or replay and the Q4 context probe arguments')
    import torch
    from dllm.device import selected_accelerator, selected_device
    if selected_accelerator() not in ('cuda', 'rocm'):
        parser.error('prefill tracing requires the CUDA/ROCm Q4 context probe')
    device = selected_device()

    def memory():
        return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                    reserved_bytes=torch.cuda.memory_reserved(device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))

    # Exclusive creation prevents accidentally replacing prior evidence.
    with open(args.trace, 'x', encoding='utf-8') as stream:
        trace = Trace(stream, lambda: torch.cuda.synchronize(device), memory)
        trace.emit(dict(event='metadata', schema_version=1, torch_version=torch.__version__,
                        device=str(device), device_name=torch.cuda.get_device_name(device),
                        timing_policy='synchronized inclusive spans; diagnostic, not benchmark',
                        memory_policy='allocator bytes; peaks are process lifetime, not per span'))
        original_argv = sys.argv
        try:
            sys.argv = ['hybrid_q4_context_probe'] + args.probe_args
            with trace_prefill(trace):
                runpy.run_module('scripts.experiments.hybrid_q4_context_probe', run_name='__main__')
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    main()
