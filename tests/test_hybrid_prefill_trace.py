"""Diagnostic event ordering and hook cleanup; no GPU performance claims."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.experiments.hybrid_prefill_trace import Trace, trace_prefill


class PrefillTraceTests(unittest.TestCase):
    def test_begin_survives_execution_failure_and_has_parent(self):
        output = io.StringIO()
        trace = Trace(output, lambda: None, lambda: {'allocated_bytes': 10})
        def fail():
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[-1]['event'], 'begin')
            raise RuntimeError('GPU failed')
        with self.assertRaises(RuntimeError):
            trace.call('chunk', lambda: trace.call('attention', fail, {}), {})
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['event'] for e in events], ['begin', 'begin', 'error', 'error'])
        self.assertEqual(events[1]['parent'], events[0]['span'])
        self.assertIsNone(trace.parent.get())

    def test_file_is_flushed_before_work_and_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            with path.open('x') as stream:
                trace = Trace(stream, lambda: None, lambda: {})
                def fail():
                    self.assertEqual(json.loads(path.read_text())['event'], 'begin')
                    raise ValueError('failed work')
                with self.assertRaises(ValueError):
                    trace.call('projection', fail, {})
                self.assertEqual(json.loads(path.read_text().splitlines()[-1])['event'], 'error')

    def test_completion_sync_failure_is_not_reported_as_success(self):
        output = io.StringIO()
        calls = []
        def synchronize():
            calls.append('sync')
            if len(calls) == 2:
                raise RuntimeError('asynchronous failure')
        trace = Trace(output, synchronize, lambda: {})
        with self.assertRaises(RuntimeError):
            trace.call('attention', lambda: None, {})
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['event'] for e in events], ['begin', 'error'])
        self.assertEqual(events[-1]['phase'], 'synchronize_after')

    def test_sync_failure_retains_begin_without_executing(self):
        output = io.StringIO()
        def synchronize():
            raise RuntimeError('sync failed')
        trace = Trace(output, synchronize, lambda: self.fail('memory called'))
        with self.assertRaises(RuntimeError):
            trace.call('chunk', lambda: self.fail('executed'), {})
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[1]['phase'], 'synchronize_before')
        self.assertIsNone(events[1]['seconds'])

    def test_prefill_only_preserves_results_and_restores_after_failure(self):
        output = io.StringIO()
        trace = Trace(output, lambda: None, lambda: {'allocated_bytes': 10})
        module = SimpleNamespace(scaled_attention=lambda q, k, v, **kw: q)
        class Executor:
            def __init__(self):
                self.w = {'0.qkv': object()}
                self.linear = lambda x, w: x
            def attention(self, x, i, *args):
                return module.scaled_attention(self.linear(x, self.w['0.qkv']), x, x)
            linear_attention = attention
            def forward_embeddings(self, x, positions, slots, prefill, prefix_length=0, return_logits=True):
                result = self.attention(x, 0)
                if prefix_length == 99:
                    raise ValueError('failed chunk')
                return result
        executor = Executor()
        x = SimpleNamespace(shape=(1, 127, 8))
        original_forward, original_attention, original_linear = Executor.forward_embeddings, module.scaled_attention, executor.linear
        with trace_prefill(trace, Executor, module):
            self.assertIs(executor.forward_embeddings(x, None, None, False), x)
            self.assertEqual(output.getvalue(), '')
            self.assertIs(executor.forward_embeddings(x, None, None, True, 254), x)
            self.assertIs(executor.linear, original_linear)
            with self.assertRaises(ValueError):
                executor.forward_embeddings(x, None, None, True, 99)
            self.assertIs(executor.linear, original_linear)
            count = len(output.getvalue())
            executor.forward_embeddings(x, None, None, False)
            self.assertEqual(len(output.getvalue()), count)
        self.assertIs(Executor.forward_embeddings, original_forward)
        self.assertIs(module.scaled_attention, original_attention)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        begins = [e for e in events if e['event'] == 'begin']
        self.assertEqual([e['operation'] for e in begins[:4]],
                         ['prefill_chunk', 'full_attention', 'projection', 'scaled_attention'])
        self.assertEqual(begins[2]['weight'], '0.qkv')
        self.assertEqual(begins[0]['prefix_tokens'], 254)
        self.assertEqual(begins[0]['chunk_tokens'], 127)
        ends = [e for e in events if e['event'] == 'end']
        self.assertEqual(ends[0]['memory_after']['allocated_bytes'], 10)
        self.assertGreaterEqual(ends[0]['seconds'], 0)
