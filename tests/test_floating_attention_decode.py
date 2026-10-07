from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

import torch

from dllm.floating_attention import attention_prefill, attention_decode
from tests.test_floating_attention import fixture


class FloatingAttentionDecodeTests(unittest.TestCase):
    def test_every_decode_step_matches_fresh_prefill_and_keeps_storage(self):
        for ratio, length, prefix in ((0,13,3),(4,13,3),(128,132,126)):
            hidden, weights, freq = fixture(ratio,length)
            _, state = attention_prefill(hidden[:,:prefix], weights, freq, groups=2, window=4,
                                         ratio=ratio, index_top_k=2, max_tokens=length)
            storage = state.cache.data_ptr()
            for position in range(prefix,length):
                actual = attention_decode(hidden[:,position:position+1], state, position=position)
                expected, reference = attention_prefill(hidden[:,:position+1], weights, freq, groups=2, window=4,
                                                        ratio=ratio, index_top_k=2)
                torch.testing.assert_close(actual, expected[:,-1:], rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(state.window, reference.window, rtol=3e-5, atol=1e-6)
                self.assertEqual(state.cache.data_ptr(), storage)
                self.assertEqual(state.position, position+1)
                if ratio:
                    torch.testing.assert_close(state.compressed[:,:state.position//ratio], reference.compressed,
                                               rtol=3e-5, atol=1e-6)

    def test_reordered_and_shrinking_slots_match_fresh_prefill(self):
        hidden, weights, freq = fixture(4,9)
        hidden = torch.cat((hidden, hidden*.8), 0)
        _, state = attention_prefill(hidden[:,:5], weights, freq, groups=2,window=4,ratio=4,max_tokens=9)
        state.reorder(torch.tensor([1,0,1]))
        reordered = hidden[[1,0,1]]
        actual = attention_decode(reordered[:,5:6], state, position=5)
        expected, _ = attention_prefill(reordered[:,:6], weights, freq, groups=2,window=4,ratio=4)
        torch.testing.assert_close(actual,expected[:,-1:],rtol=3e-5,atol=3e-6)
        state.reorder(torch.tensor([2]))
        actual = attention_decode(reordered[2:3,6:7], state, position=6)
        expected, _ = attention_prefill(reordered[2:3,:7], weights, freq, groups=2,window=4,ratio=4)
        torch.testing.assert_close(actual,expected[:,-1:],rtol=3e-5,atol=3e-6)

    def test_mutation_failure_poisons_and_release_is_idempotent(self):
        hidden, weights, freq = fixture(4,9)
        _, state = attention_prefill(hidden[:,:3], weights, freq, groups=2,window=4,ratio=4,max_tokens=9)
        with patch('dllm.floating_attention.selected_attention', side_effect=RuntimeError('attention failed')):
            with self.assertRaisesRegex(RuntimeError, 'attention failed'):
                attention_decode(hidden[:,3:4],state,position=3)
        self.assertTrue(state.poisoned)
        self.assertEqual(state.position,3)
        with self.assertRaisesRegex(RuntimeError,'poisoned'):
            attention_decode(hidden[:,3:4],state,position=3)
        with self.assertRaisesRegex(RuntimeError,'poisoned'):
            state.reorder(torch.tensor([0]))
        state.close(); state.close()
        self.assertIsNone(state.cache)
        self.assertIsNone(state.index_keys)
        self.assertIsNone(state.pool)
        self.assertFalse(state.weights)
        with self.assertRaisesRegex(RuntimeError,'closed'):
            attention_decode(hidden[:,3:4],state,position=3)

    def test_competing_updates_serialize_and_reject_stale_position(self):
        from dllm import floating_attention
        hidden, weights, freq = fixture(4,9)
        _, state = attention_prefill(hidden[:,:3],weights,freq,groups=2,window=4,ratio=4,max_tokens=9)
        entered, release = threading.Event(), threading.Event()
        original = floating_attention.selected_attention
        def hold(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test did not release attention')
            return original(*args, **kwargs)
        with patch('dllm.floating_attention.selected_attention', side_effect=hold), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(attention_decode, hidden[:,3:4], state, position=3)
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(attention_decode, hidden[:,3:4], state, position=3)
            finally:
                release.set()
            first.result(timeout=3)
            with self.assertRaises(ValueError):
                second.result(timeout=3)
        self.assertEqual(state.position,4)
        self.assertFalse(state.poisoned)

    def test_preflight_failures_preserve_state_and_budget_fails_before_projection(self):
        hidden, weights, freq = fixture(4,9)
        with patch('dllm.floating_attention.linear', side_effect=AssertionError('projection started')):
            with self.assertRaises(ValueError):
                attention_prefill(hidden,weights,freq,groups=2,window=4,ratio=4,max_cache_bytes=1)
        _, state = attention_prefill(hidden[:,:3],weights,freq,groups=2,window=4,ratio=4,max_tokens=4)
        before = state.cache.clone()
        for position in (2,4,True):
            with self.assertRaises(ValueError):
                attention_decode(hidden[:,3:4],state,position=position)
        with self.assertRaises(ValueError):
            attention_decode(hidden[:,3:5],state,position=3)
        torch.testing.assert_close(state.cache,before)
        self.assertFalse(state.poisoned)
        with patch('dllm.floating_attention.linear', side_effect=RuntimeError('preparation failed')):
            with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
                attention_decode(hidden[:,3:4],state,position=3)
        torch.testing.assert_close(state.cache,before)
        self.assertFalse(state.poisoned)
        attention_decode(hidden[:,3:4],state,position=3)
        with self.assertRaises(ValueError):
            attention_decode(hidden[:,4:5],state,position=4)
        self.assertFalse(state.poisoned)
        previous = state.cache
        state.max_cache_bytes = state.bytes_per_slot
        with self.assertRaises(ValueError):
            state.reorder(torch.tensor([0,0]))
        self.assertIs(state.cache,previous)
