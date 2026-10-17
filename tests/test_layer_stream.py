"""CPU ownership and arithmetic tests; no host/GPU bandwidth or speed claims."""
import unittest
from unittest.mock import Mock, patch

import torch

from damendaemon.layer_stream import LayerWeightStore, plan, validate
from tests import test_hybrid_eager_chunks


class LayerStreamTests(unittest.TestCase):
    def weights(self):
        embed = torch.ones(3, 4)
        return {'embed': embed, 'head': embed, 'norm': torch.ones(4),
                '0.a': torch.ones(4, 4), '0.b': torch.ones(4),
                '1.a': torch.ones(8, 4), '1.b': torch.ones(8)}

    def test_budget_aliases_and_release_before_next_copy(self):
        weights = self.weights()
        layout = plan(weights, 2)
        self.assertEqual(layout.fixed_bytes, 64)
        self.assertEqual(layout.layer_bytes, (80, 160))
        self.assertEqual(layout.peak_bytes, 224)
        events = []
        store = LayerWeightStore(weights, layout, lambda tensor: tensor.clone(),
                                 lambda: events.append('sync'), lambda: events.append('empty'))
        self.assertIs(store.resident['embed'], store.resident['head'])
        store.activate(0)
        self.assertEqual(store.resident_bytes, 144)
        transfer_count = store.transfers
        store.activate(0)
        self.assertEqual(store.transfers, transfer_count)
        def copy(tensor):
            self.assertEqual(events, ['sync', 'empty'])
            self.assertNotIn('0.a', store.resident)
            return tensor.clone()
        store.copy = copy
        store.activate(1)
        self.assertEqual(store.resident_bytes, 224)
        self.assertEqual(store.peak_live_bytes, layout.peak_bytes)
        store.close()
        self.assertEqual(store.resident_bytes, 0)
        self.assertEqual(store.source, {})
        self.assertTrue(store.closed)
        events_before = list(events)
        store.close()
        self.assertEqual(events, events_before)
        with self.assertRaises(RuntimeError):
            store.activate(0)

    def test_copy_failure_drops_partial_layer_and_poisoned_store(self):
        weights = self.weights()
        store = LayerWeightStore(weights, plan(weights, 2), lambda t: t.clone(), Mock(), Mock())
        store.activate(0)
        calls = 0
        def fail_second(tensor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError('injected allocation failure')
            return tensor.clone()
        store.copy = fail_second
        with self.assertRaises(MemoryError):
            store.activate(1)
        self.assertTrue(store.failed)
        self.assertIsNone(store.active)
        self.assertEqual(set(store.resident), {'embed', 'head', 'norm'})
        with self.assertRaises(RuntimeError):
            store.activate(0)
        store.close()
        self.assertFalse(store.resident)

    def test_failed_sync_keeps_owned_buffers_until_successful_close(self):
        weights = self.weights()
        store = LayerWeightStore(weights, plan(weights, 2), lambda t: t.clone(), Mock(), Mock())
        store.activate(0)
        store.synchronize = Mock(side_effect=RuntimeError('injected device failure'))
        with self.assertRaises(RuntimeError):
            store.activate(1)
        self.assertTrue(store.failed)
        self.assertIn('0.a', store.resident)
        with self.assertRaises(RuntimeError):
            store.close()
        self.assertFalse(store.closed)
        store.synchronize = Mock()
        store.close()
        self.assertTrue(store.closed)

    def test_plan_rejects_incomplete_or_nonfloating_sources(self):
        for change in ({'0.a': torch.ones(4, dtype=torch.int8)}, {'2.a': torch.ones(4)},
                       {'other': torch.ones(4)}):
            with self.assertRaises(ValueError):
                plan({**self.weights(), **change}, 2)
        with self.assertRaises(ValueError):
            plan(self.weights(), 3)

    def test_policy_rejects_graphs_packing_other_devices_and_precision(self):
        config = {'model_type': 'qwen3_5_text', 'max_position_embeddings': 32768}
        env = {'DAMENDAEMON_DEVICE': 'cuda', 'DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD': '1', 'DAMENDAEMON_KERNELS': 'torch'}
        with patch.dict('os.environ', env, clear=True):
            validate(config, 'none', 2048, 4)
            for overrides in ({'DAMENDAEMON_DEVICE':'rocm'}, {'DAMENDAEMON_DEVICE':'xpu'}, {'DAMENDAEMON_KERNELS':'auto'},
                              {'DAMENDAEMON_HYBRID_PRECISION':'float32'}, {'DAMENDAEMON_EXPERIMENTAL_VISION':'1'}):
                with patch.dict('os.environ', overrides), self.assertRaises(ValueError):
                    validate(config, 'none', 512)
            with self.assertRaises(ValueError):
                validate(config, 'none', 512, use_graph=True)
            with self.assertRaises(ValueError):
                validate(config, 'q8_0', 512)

    def test_real_eager_arithmetic_matches_resident_weights(self):
        # Reuse the small real hybrid tensor fixture, never a GPU mock benchmark.
        fixture = test_hybrid_eager_chunks.HybridEagerChunkTests()
        fixture.setUp()
        try:
            from damendaemon.hybrid import HybridExecutor
            from damendaemon.capacity import hybrid_state_bytes
            model = fixture.model(fp32=False)
            with patch.dict('os.environ', {'DAMENDAEMON_DEVICE':'cuda', 'DAMENDAEMON_KERNELS':'torch'}, clear=True), \
                    patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                    patch('damendaemon.hybrid.empty_cache'), patch('damendaemon.hybrid.synchronize'), \
                    patch('damendaemon.hybrid.available_memory', return_value=2**40):
                resident = HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')
                with patch.dict('os.environ', DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD='1'):
                    streamed = HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')
            store = streamed.offload
            expected_head_workspace = model.weights['head'].numel()*4
            self.assertEqual(streamed.output_head_workspace_bytes, expected_head_workspace)
            self.assertEqual(streamed.required_memory_bytes, store.layout.peak_bytes +
                             hybrid_state_bytes(model.config, 32, 2, 2) + 512*1024**2 + expected_head_workspace)
            for prompt in ([1, 2, 3, 4], [5, 6]):
                torch.testing.assert_close(streamed.prefill(prompt), resident.prefill(prompt), rtol=0, atol=0)
                for token in (7, 8):
                    torch.testing.assert_close(streamed.decode(token), resident.decode(token), rtol=0, atol=0)
                fixture.assert_state(resident, streamed)
                self.assertLessEqual(store.peak_live_bytes, store.layout.peak_bytes)
            # CUDA selection still uses the actual eager CPU tensor test: no
            # CUDA APIs or hardware are substituted for a qualification result.
            with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda'}):
                self.assertTrue(streamed.supports_chunked_prefill)
                self.assertFalse(streamed.requires_full_prefill)
                prompt = [1, 3, 5, 7, 9, 11]
                streamed.prefill_chunk(prompt[:2], 0, 0, False)
                actual = streamed.prefill_chunk(prompt[2:], 0, 2)
                torch.testing.assert_close(actual, resident.prefill(prompt), rtol=2e-3, atol=2e-3)
                fixture.assert_state(resident, streamed)
                untouched = {i: value[0].clone() for i, value in streamed.state.items()}
                streamed.prefill_chunk([2, 4], 1, 0, False)
                streamed.prefill_chunk([6], 1, 2)
                for i, value in untouched.items():
                    torch.testing.assert_close(streamed.state[i][0], value, rtol=0, atol=0)
            streamed.close()
            self.assertEqual(store.resident_bytes, 0)
        finally:
            fixture.doCleanups()

    def test_meta_headers_can_plan_but_cannot_execute(self):
        weights = {name: tensor.to('meta') for name, tensor in self.weights().items()}
        weights['head'] = weights['embed']
        layout = plan(weights, 2)
        self.assertEqual(layout.peak_bytes, 224)
        copy = Mock()
        with self.assertRaisesRegex(ValueError, 'actual CPU'):
            LayerWeightStore(weights, layout, copy, Mock(), Mock())
        copy.assert_not_called()
