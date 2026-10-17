"""Fused device-resident composition: no host sync, packed-decode equivalence.

Local CPU tests exercise the whole fused pipeline with real packed tile decoding
(on CPU tensors) against a dense-decode control. The shared-A2 fixture proves
CUDA tensors separately.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from damendaemon.cuda_packed_moe import _accumulate, _accumulate_batched, _accumulate_fused


class FakeStackedExpert:
    """Packed expert exposing weight/scales for stacking plus dense-row helpers."""

    def __init__(self, raw, scales, format, activation_limit=2):
        self.gate = SimpleNamespace(weight=raw, scales=scales, format=format)
        self.up = SimpleNamespace(weight=raw, scales=scales, format=format)
        self.down = SimpleNamespace(weight=raw.T.contiguous(), scales=scales.T.contiguous(), format=format)
        from damendaemon.packed_tiles import decode_tile
        self.width = raw.shape[1]
        self.activation_limit = activation_limit
        self._decoded = [decode_tile(raw, scales, format=format, row_start=0,
                                     row_stop=raw.shape[0], column_start=0,
                                     column_stop=raw.shape[1]) for raw in
                         (raw, raw, raw.T.contiguous())]

    def _row(self, hidden, coefficient):
        from torch.nn import functional as F
        gate = (hidden @ self._decoded[0].T).clamp(max=self.activation_limit)
        up = (hidden @ self._decoded[1].T).clamp(-self.activation_limit, self.activation_limit)
        activated = F.silu(gate)*up
        if coefficient is not None:
            activated = activated*coefficient
        return (activated @ self._decoded[2].T)

    def forward_batched(self, hidden, *, route_weights=None):
        output = torch.empty_like(hidden)
        for row in range(hidden.shape[0]):
            coefficient = None if route_weights is None else route_weights[row:row+1]
            output[row:row+1] = self._row(hidden[row:row+1], coefficient)
        return output


class FusedCompositionTests(unittest.TestCase):
    def bank(self, tokens_hidden=None, format='e4m3fn', experts=3, seed=482):
        from damendaemon.packed_tiles import decode_tile
        rng = torch.Generator().manual_seed(seed)
        rows, columns = 32, 8
        bank = []
        for index in range(experts):
            raw = (torch.rand(rows, columns, generator=rng)*119).to(torch.uint8)
            scales = torch.full((rows//32, 1), 125, dtype=torch.uint8)
            bank.append(FakeStackedExpert(raw, scales, format))
        hidden = (torch.randn(tokens_hidden, columns, generator=rng)*.2) if tokens_hidden else torch.zeros(0, columns)
        ids = torch.randint(0, experts, (tokens_hidden, 2), generator=rng) if tokens_hidden else torch.zeros(0, 2, dtype=torch.int64)
        weights = torch.rand(tokens_hidden, 2, generator=rng) if tokens_hidden else torch.zeros(0, 2)
        return bank, hidden.to(torch.float16).float(), ids, weights

    def test_fused_matches_grouped_controls(self):
        for rows in (4, 0):
            with self.subTest(rows=rows):
                bank, hidden, ids, weights = self.bank(rows)
                for shared in (None, bank[0]):
                    control = _accumulate(hidden, bank, ids, weights, shared)
                    batched = _accumulate_batched(hidden, bank, ids, weights, shared)
                    fused = _accumulate_fused(hidden, bank, ids, weights, shared)
                    torch.testing.assert_close(fused, control, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(fused, batched, rtol=1e-5, atol=1e-6)

    def test_fused_rejects_mixed_formats_and_oversized_gather(self):
        bank, hidden, ids, weights = self.bank(2)
        bank[1].gate.format = 'e2m1'
        with self.assertRaisesRegex(ValueError, 'one packed format'):
            _accumulate_fused(hidden, bank, ids, weights, None)
        bank2, hidden2, ids2, weights2 = self.bank(2)
        with self.assertRaisesRegex(ValueError, 'gathered-tile workspace budget'):
            _accumulate_fused(hidden2, bank2, ids2, weights2, None, max_gather_bytes=64)

    def test_fused_has_no_host_readback(self):
        bank, hidden, ids, weights = self.bank(3)
        with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')), \
                patch('torch.Tensor.tolist', side_effect=AssertionError('host readback')):
            _accumulate_fused(hidden, bank, ids, weights, None)


if __name__ == '__main__':
    unittest.main()
