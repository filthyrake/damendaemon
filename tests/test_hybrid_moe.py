"""CPU property tests for the qwen3.5-MoE hybrid executor surface (#668).

Real CPU tensor arithmetic on a synthetic hybrid MoE checkpoint: the FFN of
every layer is a routed top-k expert mixture plus a sigmoid-gated shared
expert (the transcribed Qwen3.6-35B-A3B surface), with the dense hybrid
attention/linear schedule unchanged. No GPU execution or quality claim.

Pinned parity: the executor's MoE FFN must match the independent host
reference ``HybridMoeFeedForward`` (hybrid_moe_ffn, semantics pinned to the
eager Transformers reference) on identical FP32 weights and inputs.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from damendaemon.config import decoder_config
from damendaemon.hybrid import HybridExecutor, load_weights
from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward

HIDDEN, HEADS, KV, HEAD_DIM = 32, 2, 1, 16
LAYERS, EXPERTS, TOP, INTER, SHARED = 4, 4, 2, 32, 32


def moe_root_config():
    return {'model_type': 'qwen3_5_moe', 'tie_word_embeddings': True,
            'image_token_id': 99, 'text_config': {
        'model_type': 'qwen3_5_moe_text', 'hidden_size': HIDDEN,
        'num_hidden_layers': LAYERS, 'num_attention_heads': HEADS,
        'num_key_value_heads': KV, 'head_dim': HEAD_DIM, 'vocab_size': 64,
        'max_position_embeddings': 262144,
        'linear_num_key_heads': 1, 'linear_num_value_heads': 2,
        'linear_key_head_dim': 16, 'linear_value_head_dim': 16,
        'linear_conv_kernel_dim': 4,
        'num_experts': EXPERTS, 'num_experts_per_tok': TOP,
        'moe_intermediate_size': INTER, 'shared_expert_intermediate_size': SHARED,
        'mtp_num_hidden_layers': 1, 'mtp_use_dedicated_embeddings': False,
        'layer_types': ['linear_attention']*3+['full_attention'],
        'hidden_act': 'silu', 'attn_output_gate': True, 'rms_norm_eps': 1.e-6,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000000.,
                            'partial_rotary_factor': .25}}}


def moe_tensors(rng):
    shapes = {'embed_tokens.weight': (64, HIDDEN), 'norm.weight': (HIDDEN,)}
    for i in range(LAYERS):
        shapes.update({f'layers.{i}.'+name: shape for name, shape in {
            'input_layernorm.weight': (HIDDEN,), 'post_attention_layernorm.weight': (HIDDEN,),
            'mlp.gate.weight': (EXPERTS, HIDDEN),
            'mlp.experts.gate_up_proj': (EXPERTS, 2*INTER, HIDDEN),
            'mlp.experts.down_proj': (EXPERTS, HIDDEN, INTER),
            'mlp.shared_expert.gate_proj.weight': (SHARED, HIDDEN),
            'mlp.shared_expert.up_proj.weight': (SHARED, HIDDEN),
            'mlp.shared_expert.down_proj.weight': (HIDDEN, SHARED),
            'mlp.shared_expert_gate.weight': (1, HIDDEN)}.items()})
        if i < LAYERS - 1:
            shapes.update({'layers.%d.linear_attn.' % i+name: shape for name, shape in {
                'in_proj_qkv.weight': (64, HIDDEN), 'in_proj_z.weight': (32, HIDDEN),
                'in_proj_b.weight': (2, HIDDEN), 'in_proj_a.weight': (2, HIDDEN),
                'conv1d.weight': (64, 1, 4), 'out_proj.weight': (HIDDEN, HIDDEN),
                'dt_bias': (2,), 'A_log': (2,), 'norm.weight': (16,)}.items()})
        else:
            shapes.update({'layers.%d.self_attn.' % i+name: shape for name, shape in {
                'q_proj.weight': (2*HEADS*HEAD_DIM, HIDDEN),
                'k_proj.weight': (KV*HEAD_DIM, HIDDEN), 'v_proj.weight': (KV*HEAD_DIM, HIDDEN),
                'o_proj.weight': (HIDDEN, HEADS*HEAD_DIM),
                'q_norm.weight': (HEAD_DIM,), 'k_norm.weight': (HEAD_DIM,)}.items()})
    return {'model.language_model.'+name: torch.randn(shape, generator=rng)*.08
            for name, shape in shapes.items()}


def mtp_tensors(rng):
    """Raw mtp.* tensors on the synthetic snapshot (the transcribed Qwen3.6
    MTP surface: fc over two pre-normed inputs, one full-attention MoE layer,
    final norm; embeddings and head shared with the main model)."""
    shapes = {'fc.weight': (HIDDEN, 2*HIDDEN),
              'pre_fc_norm_embedding.weight': (HIDDEN,),
              'pre_fc_norm_hidden.weight': (HIDDEN,), 'norm.weight': (HIDDEN,),
              'layers.0.input_layernorm.weight': (HIDDEN,),
              'layers.0.post_attention_layernorm.weight': (HIDDEN,)}
    shapes.update({'layers.0.self_attn.'+name: shape for name, shape in {
        'q_proj.weight': (2*HEADS*HEAD_DIM, HIDDEN),
        'k_proj.weight': (KV*HEAD_DIM, HIDDEN), 'v_proj.weight': (KV*HEAD_DIM, HIDDEN),
        'o_proj.weight': (HIDDEN, HEADS*HEAD_DIM),
        'q_norm.weight': (HEAD_DIM,), 'k_norm.weight': (HEAD_DIM,)}.items()})
    shapes.update({'layers.0.mlp.'+name: shape for name, shape in {
        'gate.weight': (EXPERTS, HIDDEN),
        'experts.gate_up_proj': (EXPERTS, 2*INTER, HIDDEN),
        'experts.down_proj': (EXPERTS, HIDDEN, INTER),
        'shared_expert.gate_proj.weight': (SHARED, HIDDEN),
        'shared_expert.up_proj.weight': (SHARED, HIDDEN),
        'shared_expert.down_proj.weight': (HIDDEN, SHARED),
        'shared_expert_gate.weight': (1, HIDDEN)}.items()})
    return {'mtp.'+name: torch.randn(shape, generator=rng)*.08
            for name, shape in shapes.items()}


class HybridMoeTests(unittest.TestCase):
    def setUp(self):
        # CPU execution env (the Intel bring-up policy deliberately rejects the
        # unqualified MoE type; XPU cells wait for their own qualification).
        self.env = patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'xpu',
                                             'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                                             'DAMENDAEMON_EXPERIMENTAL_VISION': '0'},
                              clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = decoder_config(moe_root_config())
        rng = torch.Generator().manual_seed(123)
        self.state = moe_tensors(rng)
        self.weights = load_weights(self.state, self.config)

    def model(self):
        weights = {name: value.float() for name, value in self.weights.items()}
        weights['head'] = weights['embed']
        return SimpleNamespace(config=self.config, max_context=64, weights=weights,
                               scales={}, quantization='none',
                               bytes=sum(t.numel()*t.element_size() for t in
                                         {id(t):t for t in weights.values()}.values()))

    def executor(self):
        # CPU execution through the float32 precision path (the house fixture
        # pattern): the device boundary is patched wholesale, including the
        # Intel bring-up gate that deliberately rejects the unqualified MoE
        # type — XPU cells wait for their own qualification.
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.xpu_policy.validate_model', lambda *args, **kwargs: None), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(self.model(), use_graph=False, max_batch=2,
                                  kernels='torch')

    # ── config admission ────────────────────────────────────────────

    def test_moe_text_config_decodes_with_expert_geometry(self):
        c = self.config
        self.assertEqual(c['model_type'], 'qwen3_5_moe_text')
        self.assertEqual((c['num_experts'], c['num_experts_per_tok']), (EXPERTS, TOP))
        self.assertEqual(c['moe_intermediate_size'], INTER)
        self.assertEqual(c['shared_expert_intermediate_size'], SHARED)
        self.assertNotIn('intermediate_size', c)

    def test_dense_text_config_still_rejects_experts(self):
        from tests.test_config import HybridConfigTests
        config = HybridConfigTests().config()
        config['text_config']['num_experts'] = 4
        with self.assertRaises(ValueError):
            decoder_config(config)

    def test_moe_config_requires_complete_expert_geometry(self):
        for key in ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                    'shared_expert_intermediate_size'):
            config = moe_root_config()
            del config['text_config'][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                decoder_config(config)

    # ── load path ───────────────────────────────────────────────────

    def test_load_weights_carries_the_transcribed_moe_surface(self):
        for i in range(LAYERS):
            for name, shape in ((f'{i}.router', (EXPERTS, HIDDEN)),
                                (f'{i}.experts_gate_up', (EXPERTS, 2*INTER, HIDDEN)),
                                (f'{i}.experts_down', (EXPERTS, HIDDEN, INTER)),
                                (f'{i}.shared_gate_up', (2*SHARED, HIDDEN)),
                                (f'{i}.shared_down', (HIDDEN, SHARED)),
                                (f'{i}.shared_expert_gate', (1, HIDDEN))):
                self.assertEqual(tuple(self.weights[name].shape), shape)
            self.assertNotIn(f'{i}.gate_up', self.weights)
        self.assertFalse(self.state, 'every tensor must be consumed')

    def test_moe_state_with_dense_ffn_tensors_rejects(self):
        self.state['model.language_model.layers.0.mlp.gate_proj.weight'] = \
            torch.randn(INTER, HIDDEN)
        with self.assertRaises(ValueError):
            load_weights(self.state, self.config)

    # ── pinned parity vs the independent host reference ─────────────

    def test_moe_ffn_matches_host_reference_exactly(self):
        executor = self.executor()
        rng = torch.Generator().manual_seed(7)
        reference = HybridMoeFeedForward(
            self.weights['1.router'].float(), self.weights['1.experts_gate_up'].float(),
            self.weights['1.experts_down'].float(),
            self.weights['1.shared_gate_up'][:SHARED].float(),
            self.weights['1.shared_gate_up'][SHARED:].float(),
            self.weights['1.shared_down'].float(),
            self.weights['1.shared_expert_gate'].float(), top_k=TOP)
        x = torch.randn(1, 5, HIDDEN, generator=rng)
        # The executor applies norm2 inside _moe_half; the reference consumes
        # the post-norm hidden exactly like the eager MLP block.
        normed = executor.norm(x, executor.w['1.norm2']).float()
        expected = reference.forward(normed)
        actual = executor._moe_half(x, 1) - x  # residual added by the sublayer
        torch.testing.assert_close(actual.reshape(normed.shape), expected, rtol=2e-6, atol=2e-6)

    # ── execution properties ────────────────────────────────────────

    def test_prefill_and_decode_run_through_the_moe_path(self):
        executor = self.executor()
        prompt = [1, 4, 2, 7, 5, 3, 8]
        logits = executor.prefill(prompt)
        self.assertTrue(torch.isfinite(logits).all())
        token = int(logits.argmax(-1)[0])
        for _ in range(3):
            logits, next_ids = executor.decode(token)
            self.assertTrue(torch.isfinite(logits).all())
            token = int(next_ids[0])

    def test_chunked_prefill_equals_whole_prefill(self):
        prompt = [1, 4, 2, 7, 5, 3, 8, 6, 9, 11]
        whole, chunked = self.executor(), self.executor()
        target = whole.prefill(prompt)
        offset, actual = 0, None
        for index, size in enumerate((4, 4, 3)):
            final = index == 2
            actual = chunked.prefill_chunk(prompt[offset:offset+size], 0, offset, final)
            offset += size
        torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
        token = int(actual.argmax(-1)[0])
        for _ in range(2):
            _, whole_next = whole.decode(token)
            _, chunked_next = chunked.decode(token)
            self.assertEqual(int(whole_next[0]), int(chunked_next[0]))
            token = int(chunked_next[0])

    # ── packed expert projections (#668 cell 2) ─────────────────────

    def _prepared(self, quantization):
        import copy
        weights = {name: value.clone() for name, value in self.weights.items()}
        weights['head'] = weights['embed']
        from damendaemon.quantization import prepare
        scales = prepare(weights, quantization)
        return weights, scales

    def test_prepare_packs_expert_stacks_rowwise_and_keeps_routing_exact(self):
        for quantization in ('q8_0', 'q4_0'):
            with self.subTest(quantization=quantization):
                weights, scales = self._prepared(quantization)
                packed_dtype = torch.int8 if quantization == 'q8_0' else torch.uint8
                self.assertEqual(weights['1.experts_gate_up'].dtype, packed_dtype)
                self.assertEqual(weights['1.experts_down'].dtype, packed_dtype)
                self.assertEqual(weights['1.experts_gate_up'].shape,
                                 (EXPERTS, 2*INTER, HIDDEN if quantization == 'q8_0' else HIDDEN//2))
                self.assertEqual(scales['1.experts_gate_up'].shape, (EXPERTS, 2*INTER, HIDDEN//32))
                # Routers and combine gates are never packed.
                for name in ('1.router', '1.shared_expert_gate'):
                    self.assertNotIn(name, scales)
                    self.assertNotEqual(weights[name].dtype, packed_dtype)
                # Shared expert projections pack as ordinary 2-D matrices.
                self.assertEqual(weights['1.shared_gate_up'].dtype, packed_dtype)
                # Every packed tensor carries exactly one scale per 32 values.
                for name, tensor in weights.items():
                    if name in scales:
                        self.assertEqual(scales[name].numel()*32,
                                         self.weights[name].numel())

    def test_packed_expert_slices_dequantize_to_their_own_rows(self):
        from damendaemon.quantization import dequantize, dequantize_q4
        # Q4_0 is 4-bit: ~10% per-weight error on synthetic randn fixtures is
        # the expected block-quantization floor, not a slicing defect (slicing
        # correctness is the row-exactness assertion inside each subtest).
        for quantization, dequant, tolerance in (('q8_0', dequantize, 0.02),
                                                 ('q4_0', dequantize_q4, 0.15)):
            with self.subTest(quantization=quantization):
                weights, scales = self._prepared(quantization)
                for stack, scale in (('1.experts_gate_up', '1.experts_gate_up'),
                                     ('1.experts_down', '1.experts_down')):
                    for expert in range(EXPERTS):
                        recovered = dequant(weights[stack][expert], scales[scale][expert],
                                            torch.float32)
                        original = self.weights[stack][expert].float()
                        error = (recovered-original).norm()/original.norm()
                        self.assertLess(float(error), tolerance)

    def test_packed_shared_projection_matches_dequantized_reference(self):
        # quantized_torch.linear (dequantize + F.linear) is the exact semantic
        # the executor's shared-expert path uses through the packed Linear for
        # Q8 (its only qualified surface; Q4's CUDA dispatch is the triton
        # kernel, verified on hardware in the A6000 cell).
        from damendaemon.quantized_torch import linear as packed_linear
        from damendaemon.quantization import dequantize, dequantize_q4
        weights, scales = self._prepared('q8_0')
        rng = torch.Generator().manual_seed(5)
        x = torch.randn(3, HIDDEN, generator=rng)
        actual = packed_linear(x, weights['1.shared_gate_up'],
                               scales['1.shared_gate_up'])
        matrix = dequantize(weights['1.shared_gate_up'],
                            scales['1.shared_gate_up'], torch.float32)
        torch.testing.assert_close(actual, torch.nn.functional.linear(x, matrix),
                                   rtol=1e-5, atol=1e-5)
        # Q4: the same dequantize + F.linear form _moe_half applies, on the
        # shared 2-D projection (determinism plus the 4-bit error floor).
        weights, scales = self._prepared('q4_0')
        matrix = dequantize_q4(weights['1.shared_gate_up'].reshape(-1, HIDDEN//2),
                               scales['1.shared_gate_up'], torch.float32)
        original = torch.cat([self.weights['1.shared_gate_up'][:SHARED].float(),
                              self.weights['1.shared_gate_up'][SHARED:].float()])
        error = (matrix-original).norm()/original.norm()
        self.assertLess(float(error), 0.15)


class HybridMoeServeAdmissionTests(unittest.TestCase):
    """The serve path dispatches MoE text through the hybrid executor (#668)."""

    def _model(self, vision):
        from types import SimpleNamespace as NS
        config = moe_root_config()['text_config']
        return NS(config=dict(config), vision_weights=NS(bytes=8) if vision else None,
                  quantization='q8_0', max_context=512)

    def test_make_executor_routes_moe_text_through_hybrid(self):
        from damendaemon import runtime
        with patch('damendaemon.hybrid.HybridExecutor') as construct:
            executor = runtime.make_executor(self._model(vision=False),
                                             use_graph=False, max_batch=1)
        self.assertIs(executor, construct.return_value)
        construct.assert_called_once()

    def test_make_executor_refuses_a_vision_moe_model(self):
        # The transcribed contract excludes the MoE snapshot's visual
        # tensors; vision weights on a MoE model are a loading defect.
        from damendaemon import runtime
        with self.assertRaisesRegex(ValueError, 'does not carry vision weights'):
            runtime.make_executor(self._model(vision=True), use_graph=False)


class HybridMoeMtpTests(unittest.TestCase):
    """The transcribed MTP draft head (#668 MTP cell).

    Load-bearing property: spec_decode's emitted stream is BY CONSTRUCTION the
    main model's greedy argmax chain — every draft is verified against the
    main head — so speculative and plain greedy must agree exactly on the
    same weights, regardless of draft quality or concat order.
    """

    def setUp(self):
        self.config = decoder_config(moe_root_config())

    def _state(self):
        state = moe_tensors(torch.Generator().manual_seed(123))
        state.update(mtp_tensors(torch.Generator().manual_seed(456)))
        return state

    def _model(self, env):
        # CPU execution env mirrors HybridMoeTests (the xpu policy forces the
        # float32 precision path; selected_device is patched to cpu below).
        environment = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                       'DAMENDAEMON_EXPERIMENTAL_VISION': '0', **env}
        with patch.dict('os.environ', environment, clear=False):
            weights = load_weights(self._state(), self.config)
        weights = {name: value.float() for name, value in weights.items()}
        weights['head'] = weights['embed']
        return SimpleNamespace(config=self.config, max_context=64, weights=weights,
                                scales={}, quantization='none',
                                bytes=sum(t.numel()*t.element_size() for t in
                                          {id(t):t for t in weights.values()}.values()))

    def _executor(self, env):
        model = self._model(env)
        # The precision policy reads env at construction — keep the CPU
        # fixture env active through HybridExecutor.__init__.
        environment = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                       'DAMENDAEMON_EXPERIMENTAL_VISION': '0', **env}
        with patch.dict('os.environ', environment, clear=False), \
                patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.xpu_policy.validate_model', lambda *a, **k: None), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')

    def test_mtp_load_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'DAMENDAEMON_HYBRID_MTP'):
            self._model({'DAMENDAEMON_HYBRID_MTP_DROP': '0'})

    def test_mtp_drop_env_skips_the_head(self):
        model = self._model({'DAMENDAEMON_HYBRID_MTP': '1',
                             'DAMENDAEMON_HYBRID_MTP_DROP': '1'})
        self.assertNotIn('mtp.fc', model.weights)
        executor = self._executor({'DAMENDAEMON_HYBRID_MTP': '1',
                                   'DAMENDAEMON_HYBRID_MTP_DROP': '1'})
        self.assertIsNone(executor.mtp_layer)

    def test_mtp_load_transcribes_the_virtual_layer(self):
        model = self._model({'DAMENDAEMON_HYBRID_MTP': '1'})
        for name in ('mtp.fc', 'mtp.pre_e', 'mtp.pre_h', 'mtp.norm',
                     f'{LAYERS}.router', f'{LAYERS}.qkv', f'{LAYERS}.o',
                     f'{LAYERS}.experts_gate_up', f'{LAYERS}.shared_expert_gate'):
            self.assertIn(name, model.weights)
        # Nothing left untranscribed.
        executor = self._executor({'DAMENDAEMON_HYBRID_MTP': '1'})
        self.assertEqual(executor.mtp_layer, LAYERS)
        self.assertIn(executor.mtp_layer, executor.kv)
        # The layer-types table is NOT extended: main forwards stay 0..N-1.
        self.assertEqual(len(executor.c['layer_types']), LAYERS)

    def test_mtp_refuses_dedicated_embeddings(self):
        config = decoder_config(moe_root_config())
        config['mtp_use_dedicated_embeddings'] = True
        state = self._state()
        environment = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                       'DAMENDAEMON_EXPERIMENTAL_VISION': '0', 'DAMENDAEMON_HYBRID_MTP': '1'}
        with patch.dict('os.environ', environment, clear=False):
            with self.assertRaisesRegex(ValueError, 'dedicated embeddings'):
                load_weights(state, config)

    def _plain_greedy(self, executor, prompt, count):
        logits = executor.prefill(prompt, 0)
        token = int(logits[0].float().argmax())
        output = [token]
        for _ in range(count - 1):
            logits, next_ids = executor.decode(token)
            token = int(next_ids[0])
            output.append(token)
        return output

    def test_spec_decode_matches_plain_greedy_exactly(self):
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        count = 24
        plain = self._executor({'DAMENDAEMON_HYBRID_MTP': '1',
                                'DAMENDAEMON_HYBRID_MTP_DROP': '1'})
        expected = self._plain_greedy(plain, prompt, count)
        for order in ('embed_first', 'hidden_first'):
            spec = self._executor({'DAMENDAEMON_HYBRID_MTP': '1',
                                   'DAMENDAEMON_HYBRID_MTP_CONCAT': order})
            output, accepted, steps = spec.spec_decode(prompt, count)
            self.assertEqual(output, expected,
                             f'spec stream diverged from plain greedy ({order})')
            self.assertEqual(len(output), count)
            self.assertLessEqual(accepted, steps)
            self.assertLessEqual(steps, count - 1)

    def test_spec_decode_requires_the_head(self):
        plain = self._executor({'DAMENDAEMON_HYBRID_MTP': '1',
                                'DAMENDAEMON_HYBRID_MTP_DROP': '1'})
        with self.assertRaisesRegex(ValueError, 'MTP draft head'):
            plain.spec_decode([1, 2, 3], 4)

    def test_mtp_concat_order_changes_draft_logits(self):
        # The fc input order is a real semantic choice (upstream drops mtp.*
        # at load, so no eager reference pins it); with random weights the
        # two orders must produce different draft logits — the A6000 gate
        # measures agreement on real weights to pin the order.
        logits = {}
        for order in ('embed_first', 'hidden_first'):
            spec = self._executor({'DAMENDAEMON_HYBRID_MTP': '1',
                                   'DAMENDAEMON_HYBRID_MTP_CONCAT': order})
            spec.prefill([1, 2, 3, 7], 0)
            hidden = spec._last_hidden
            logits[order] = spec.mtp_propose(hidden, 5, 4, 0)[0]
        self.assertFalse(torch.allclose(logits['embed_first'], logits['hidden_first']))


if __name__ == '__main__':
    unittest.main()
