import unittest
import torch
import torch.nn.functional as F
from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward

EXPERTS, DIM, INTER, SHARED, TOP_K = 4, 16, 8, 6, 2


def weights(seed=7):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32) * 0.25
    return dict(
        router=rand(EXPERTS, DIM),
        gate_up=rand(EXPERTS, 2 * INTER, DIM),
        down=rand(EXPERTS, DIM, INTER),
        shared_gate=rand(SHARED, DIM),
        shared_up=rand(SHARED, DIM),
        shared_down=rand(DIM, SHARED),
        shared_expert_gate=rand(1, DIM),
    )


def naive(w, hidden, scoring, renormalize):
    """Independent per-token control with plain torch ops."""
    out = torch.zeros_like(hidden)
    for b, t in ((b, t) for b in range(hidden.shape[0]) for t in range(hidden.shape[1])):
        x = hidden[b, t]
        logits = F.linear(x, w['router'])
        probabilities = torch.softmax(logits, -1, dtype=torch.float32) if scoring == 'softmax' \
            else logits.float().sigmoid()
        values, ids = probabilities.topk(TOP_K, dim=-1)
        if renormalize:
            values = values / values.sum()
        value = torch.zeros_like(x)
        for slot in range(TOP_K):
            gate, up = F.linear(x, w['gate_up'][ids[slot]]).chunk(2)
            expert = F.linear(F.silu(gate) * up, w['down'][ids[slot]])
            value = value + expert * values[slot]
        shared = F.linear(F.silu(F.linear(x, w['shared_gate'])) * F.linear(x, w['shared_up']),
                           w['shared_down'])
        combine = torch.sigmoid(F.linear(x, w['shared_expert_gate']))
        out[b, t] = value + shared * combine
    return out


class TestHybridMoeFeedForward(unittest.TestCase):
    def ffn(self, w=None, top_k=TOP_K, **policy):
        w = weights() if w is None else w
        return HybridMoeFeedForward(
            w['router'], w['gate_up'], w['down'], w['shared_gate'], w['shared_up'],
            w['shared_down'], w['shared_expert_gate'], top_k=top_k, **policy)

    def hidden(self, batch=2, tokens=5, seed=11):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(batch, tokens, DIM, generator=g, dtype=torch.float32) * 0.5

    def test_matches_independent_control(self):
        for scoring, renormalize in (('sigmoid', True), ('sigmoid', False),
                                     ('softmax', True), ('softmax', False)):
            with self.subTest(scoring=scoring, renormalize=renormalize):
                w, hidden = weights(), self.hidden()
                result = self.ffn(w, scoring=scoring, renormalize=renormalize).forward(hidden)
                expected = naive(w, hidden, scoring, renormalize)
                self.assertTrue(torch.allclose(result, expected, atol=1e-5, rtol=1e-5),
                                f'max deviation {(result - expected).abs().max()}')

    def test_batch_and_token_shapes(self):
        w = weights()
        for shape in ((1, 1), (1, 7), (3, 4)):
            with self.subTest(shape=shape):
                result = self.ffn(w).forward(self.hidden(*shape))
                self.assertTrue(torch.isfinite(result).all() and result.shape == shape + (DIM,))

    def test_qualified_default_policy(self):
        # Pinned to Transformers 5.20.2 Qwen3_5MoeTopKRouter: softmax +
        # unconditional top-k renormalization.
        ffn = self.ffn()
        self.assertEqual((ffn.scoring, ffn.renormalize), ('softmax', True))

    def test_policy_rejection(self):
        with self.assertRaises(ValueError): self.ffn(top_k=0)
        with self.assertRaises(ValueError): self.ffn(top_k=EXPERTS + 1)
        with self.assertRaises(ValueError): self.ffn(scoring='sqrtsoftplus')
        with self.assertRaises(ValueError): self.ffn(renormalize=1)
        with self.assertRaises(ValueError): self.ffn(workspace_bytes=0)

    def test_geometry_rejection(self):
        base = weights()
        cases = {
            'router shape': dict(router=torch.randn(EXPERTS + 1, DIM)),
            'odd fused width': dict(gate_up=torch.randn(EXPERTS, 2 * INTER + 1, DIM)),
            'down mismatch': dict(down=torch.randn(EXPERTS, DIM, INTER + 1)),
            'shared gate mismatch': dict(shared_gate=torch.randn(SHARED + 1, DIM)),
            'shared down mismatch': dict(shared_down=torch.randn(DIM, SHARED + 1)),
            'combine gate mismatch': dict(shared_expert_gate=torch.randn(2, DIM)),
            'fp64 dtype': dict(router=base['router'].double()),
        }
        for name, override in cases.items():
            with self.subTest(case=name):
                w = dict(base, **override)
                with self.assertRaises(ValueError):
                    self.ffn(w)
        with self.assertRaises(ValueError):
            HybridMoeFeedForward(base['router'][0], base['gate_up'], base['down'],
                                 base['shared_gate'], base['shared_up'], base['shared_down'],
                                 base['shared_expert_gate'], top_k=TOP_K)

    def test_nonfinite_rejection(self):
        base = weights()
        poisoned = dict(base, router=base['router'].clone())
        poisoned['router'][0, 0] = float('nan')
        with self.assertRaises(ValueError): self.ffn(poisoned)
        hidden = self.hidden(); hidden[0, 0, 0] = float('inf')
        with self.assertRaises(ValueError): self.ffn().forward(hidden)

    def test_input_rejection(self):
        with self.assertRaises(ValueError): self.ffn().forward(self.hidden().double())
        with self.assertRaises(ValueError): self.ffn().forward(self.hidden(batch=0))
        with self.assertRaises(ValueError): self.ffn().forward(torch.randn(DIM, DIM))

    def test_workspace_budget(self):
        ffn = self.ffn(workspace_bytes=1024)  # far below the real scratch bound
        with self.assertRaises(ValueError): ffn.forward(self.hidden())

    def test_output_budget(self):
        ffn = self.ffn(max_output_bytes=4 * DIM)  # one row exactly; two rows exceed
        with self.assertRaises(ValueError): ffn.forward(self.hidden(batch=1, tokens=2))
