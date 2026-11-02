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
        # Packed tensors admit structurally when a scale is supplied for the
        # same name, and the plan prices the scale with its weight (#668
        # quantized streaming): int8 '0.a' (16 bytes) + its fp16 scale
        # (32 bytes) + the unpacked fp32 '0.b' (16 bytes) = 64.
        packed = {**self.weights(), '0.a': torch.ones(4, 4, dtype=torch.int8)}
        scales = {'0.a': torch.ones(4, 4, dtype=torch.float16)}
        layout = plan(packed, 2, scales)
        self.assertEqual(layout.layer_bytes[0], 16 + 32 + 16)
        # A packed tensor whose scale is missing is not a streamable weight.
        with self.assertRaises(ValueError):
            plan({**packed, '1.a': torch.ones(8, 4, dtype=torch.int8)}, 2, scales)

    def test_policy_rejects_graphs_packing_other_devices_and_precision(self):
        config = {'model_type': 'qwen3_5_text', 'max_position_embeddings': 32768}
        env = {'DAMENDAEMON_DEVICE': 'cuda', 'DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD': '1', 'DAMENDAEMON_KERNELS': 'torch'}
        with patch.dict('os.environ', env, clear=True):
            validate(config, 'none', 2048, 4)
            # Family admission is structural: the MoE text surface shares the
            # streamed geometry and is admitted identically.
            moe = {**config, 'model_type': 'qwen3_5_moe_text', 'num_experts': 4}
            validate(moe, 'none', 2048, 4)
            for overrides in ({'DAMENDAEMON_DEVICE':'rocm'}, {'DAMENDAEMON_DEVICE':'xpu'}, {'DAMENDAEMON_KERNELS':'auto'},
                              {'DAMENDAEMON_HYBRID_PRECISION':'float32'}, {'DAMENDAEMON_EXPERIMENTAL_VISION':'1'}):
                with patch.dict('os.environ', overrides), self.assertRaises(ValueError):
                    validate(config, 'none', 512)
            with self.assertRaises(ValueError):
                validate(config, 'none', 512, use_graph=True)
            # Packed Q8/Q4 stream structurally (scales ride with their
            # weights); anything else is refused.
            validate(config, 'q8_0', 512)
            validate(moe, 'q8_0', 512)
            validate(config, 'q4_0', 512)
            with self.assertRaises(ValueError):
                validate(config, 'e5m2', 512)
            # Admission follows the family rule, not a name: a non-hybrid
            # decoder is refused whatever its name.
            with self.assertRaises(ValueError):
                validate({**config, 'model_type': 'qwen3'}, 'none', 512)

    def test_plan_partitions_head_modules_and_virtual_layers(self):
        # Head-module tensors (mtp.*) carry no layer index: fixed-resident.
        # The MTP virtual layer streams like any other layer.
        weights = {**self.weights(), 'mtp.fc': torch.ones(8, 8),
                   'mtp.norm': torch.ones(8), '2.a': torch.ones(4, 4)}
        layout = plan(weights, 3)
        self.assertEqual(set(layout.fixed), {'embed', 'head', 'norm', 'mtp.fc', 'mtp.norm'})
        self.assertEqual(layout.layer_bytes, (80, 160, 64))
        # A head module without the streaming surface is still rejected.
        with self.assertRaises(ValueError):
            plan({'mtp.fc': torch.ones(8, 8), '0.a': torch.ones(4, 4)}, 1)

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


class LayerStreamMoeTests(unittest.TestCase):
    """Streaming parity for the MoE surface (#668 family extension): the
    streamed executor must match the resident executor exactly on identical
    weights — the offload path is arithmetic-transparent by construction."""

    def test_streamed_moe_matches_resident_moe(self):
        from damendaemon.hybrid import HybridExecutor
        from tests.test_hybrid_moe import HybridMoeMtpTests
        fixture = HybridMoeMtpTests()
        fixture.setUp()
        try:
            from types import SimpleNamespace
            fp32 = fixture._model({'DAMENDAEMON_HYBRID_MTP': '1',
                                   'DAMENDAEMON_HYBRID_MTP_CONCAT': 'embed_first'})
            # Streaming is a production FP16 surface: mirror the dense
            # streaming fixture — fp16 source weights, fp16 eager compute.
            weights = {name: value.half() for name, value in fp32.weights.items()}
            weights['head'] = weights['embed']
            model = SimpleNamespace(config=fp32.config, max_context=64, weights=weights,
                                     scales={}, quantization='none',
                                     bytes=sum(t.numel()*t.element_size() for t in
                                              {id(t):t for t in weights.values()}.values()))
            base = {'DAMENDAEMON_DEVICE': 'cuda', 'DAMENDAEMON_KERNELS': 'torch',
                    'DAMENDAEMON_HYBRID_MTP': '1'}
            with patch.dict('os.environ', base, clear=False), \
                    patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                    patch('damendaemon.hybrid.empty_cache'), patch('damendaemon.hybrid.synchronize'), \
                    patch('damendaemon.hybrid.available_memory', return_value=2**40):
                resident = HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')
                with patch.dict('os.environ', DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD='1'):
                    streamed = HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')
            self.assertIsNotNone(streamed.offload)
            # The streamed peak is the fixed set plus the largest layer,
            # strictly below the fully resident footprint.
            self.assertLess(streamed.offload.layout.peak_bytes, model.bytes)
            for prompt in ([1, 2, 3, 4], [5, 6, 7]):
                torch.testing.assert_close(streamed.prefill(prompt), resident.prefill(prompt),
                                           rtol=0, atol=0)
                for token in (8, 9):
                    torch.testing.assert_close(streamed.decode(token), resident.decode(token),
                                               rtol=0, atol=0)
            # MTP composition: the draft head streams its virtual layer and
            # the spec stream still equals plain greedy exactly.
            plain_out, _, _ = streamed.spec_decode([1, 2, 3, 4], 6)
            resident_ref = fixture._plain_greedy(resident, [1, 2, 3, 4], 6)
            self.assertEqual(plain_out, resident_ref)
            streamed.close()
            self.assertEqual(streamed.offload.resident_bytes, 0)
        finally:
            fixture.doCleanups()


class LayerStreamQuantizedTests(unittest.TestCase):
    """Q8 scale streaming (#668 quantized streaming): packed weights stream
    their scales with the same alias/pop discipline as the weights, and the
    executor's rebind pattern tracks brand-new device identities after every
    activation. Full executor parity runs on the A6000 — packed eager is a
    native-GPU surface by policy."""

    def test_scales_stream_and_rebind_tracks_fresh_identities(self):
        embed = torch.ones(3, 4, dtype=torch.float16)
        weights = {'embed': embed, 'head': embed, 'norm': torch.ones(4, dtype=torch.float16),
                   '0.a': torch.ones(4, 4, dtype=torch.int8), '0.b': torch.ones(4),
                   '1.a': torch.ones(8, 4, dtype=torch.int8)}
        scales = {'head': torch.ones(3, 4, dtype=torch.float16),
                  '0.a': torch.ones(4, 4, dtype=torch.float16),
                  '1.a': torch.ones(8, 4, dtype=torch.float16)}
        layout = plan(weights, 2, scales)
        # The plan prices scales with their weights: fixed = embed+norm
        # (24+16... in fp16: embed 24, norm 16, head aliased) + head scale 24;
        # layers carry their int8 weights plus fp16 scale rows.
        self.assertEqual(layout.fixed_bytes,
                         3*4*2 + 4*2 + 3*4*2)  # embed, norm, head scale (head aliased)
        store = LayerWeightStore(weights, layout, lambda tensor: tensor.clone(),
                                lambda: None, lambda: None, scales=scales)
        # Fixed set: head's scale streams with it at construction.
        self.assertEqual(set(store.resident_scales), {'head'})
        binding = {}
        def rebind():
            # The executor's _rebind_streamed_scales pattern, verbatim.
            binding.clear()
            binding.update({id(store.resident[name]): store.resident_scales[name]
                            for name in store.resident if name in store.resident_scales})
        rebind()
        store.activate(0)
        rebind()
        self.assertEqual(set(store.resident_scales), {'head', '0.a'})
        self.assertEqual(set(binding), {id(store.resident[name])
                                       for name in ('head', '0.a') if name in store.resident_scales})

        head_identity = id(store.resident['head'])
        store.activate(1)
        rebind()
        self.assertEqual(set(store.resident_scales), {'head', '1.a'})
        # The invariant that matters: every binding key is the id of a
        # CURRENTLY resident scaled tensor — the wholesale rebind never
        # leaves entries for released tensors. (A released tensor's id can
        # be recycled by a new clone, so key-absence alone is not provable;
        # key-to-resident correspondence is.)
        expected_keys = {id(store.resident[name]) for name in store.resident_scales}
        self.assertEqual(set(binding), expected_keys)
        # Fixed tensors keep their identity by design; the resident set
        # itself proves layer 0 was released.
        self.assertNotIn('0.a', store.resident)
        self.assertIn(head_identity, binding)
        # Release drops layer scales with layer weights; accounting prices
        # both. copied_bytes is cumulative transfer volume: fixed once, each
        # activation re-copies its layer (0 twice: 16+16+32 each pass).
        store.activate(0)
        self.assertEqual(set(store.resident_scales), {'head', '0.a'})
        # The plan peak is the fixed set plus the LARGEST layer; reaching it
        # requires activating layer 1 (96 bytes vs layer 0's 64).
        store.activate(1)
        self.assertEqual(store.peak_live_bytes, layout.peak_bytes)
        expected_copied = (3*4*2 + 4*2          # embed (head aliased), norm
                           + 3*4*2              # head scale
                           + 2*(4*4 + 4*4 + 4*4*2)   # layer 0 twice: int8, fp32 b, fp16 scale
                           + 2*(8*4 + 8*4*2))   # layer 1 twice: int8 weight + fp16 scale
        self.assertEqual(store.copied_bytes, expected_copied)
        store.close()
        self.assertEqual(store.resident_scales, {})
