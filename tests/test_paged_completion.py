"""Paged terminal success must follow reservation release, using fake executors."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from dllm.scheduler import BatchedRuntime
from tests.test_scheduler import FakeRuntime, FakePagedExecutor, PagedChunkExecutor


class PagedCompletionTests(unittest.TestCase):
    def exercise(self, mode, fail):
        chunked = mode in ('chunk', 'prefix')
        with patch('dllm.scheduler.Runtime', FakeRuntime):
            runtime = BatchedRuntime({'a': 'a'}, batch_size=2, queue_size=4,
                                     prefill_chunk_size=2 if chunked else 0)
        executor = PagedChunkExecutor() if chunked else FakePagedExecutor()
        runtime.runtime.executor = executor
        executor.release.set()
        if mode == 'prefix':
            import torch
            logits = torch.zeros(1,10); logits[0,1] = 1
            executor.prefix_cache = SimpleNamespace(restore=Mock(return_value=(2, logits)), remember=Mock())
        entered, proceed = threading.Event(), threading.Event()
        original = executor.release_request
        first = True
        def release(slot):
            nonlocal first
            if first:
                first = False
                entered.set()
                if not proceed.wait(3): raise RuntimeError('test did not release cleanup')
                if fail: raise RuntimeError('injected KV release failure')
            original(slot)
        executor.release_request = release
        count = 2 if mode == 'decode' else 1
        def generate():
            return list(runtime.generate('a', [{'content': 'ab' if chunked else 'ok'}], count))
        try:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(generate)
                try:
                    self.assertTrue(entered.wait(2))
                    with self.assertRaises(FutureTimeout): future.result(.05)
                finally:
                    proceed.set()
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'KV release failure'): future.result(2)
                else:
                    self.assertEqual(future.result(2)[-1]['finish_reason'], 'length')
            runtime.manage('status')
            self.assertEqual(executor.pool.used_tokens, 0)
        finally:
            proceed.set()
            runtime.close()

    def test_terminal_waits_for_release_in_all_paged_completion_paths(self):
        for mode in ('prefill', 'decode', 'chunk', 'prefix'):
            with self.subTest(mode=mode): self.exercise(mode, False)

    def test_release_failure_replaces_terminal_success(self):
        for mode in ('prefill', 'decode', 'chunk', 'prefix'):
            with self.subTest(mode=mode): self.exercise(mode, True)
