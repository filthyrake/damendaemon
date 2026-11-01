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
LAYERS, EXPERTS, TOP, INTER, SHARED = 4, 4, 2, 16, 16


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


if __name__ == '__main__':
    unittest.main()
