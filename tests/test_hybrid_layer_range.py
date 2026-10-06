"""CPU composition tests for a shared layer boundary, not multi-GPU proof."""
import unittest

import torch

from tests import test_hybrid_eager_chunks


class HybridLayerRangeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_eager_chunks.HybridEagerChunkTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        c = self.fixture.config
        c['num_hidden_layers'] = 4
        c['layer_types'] = c['layer_types']*2
        self.fixture.weights.update({f'{int(name.split(".")[0])+2}.{name.split(".",1)[1]}': tensor.clone()
            for name, tensor in list(self.fixture.weights.items()) if name.split('.')[0].isdigit()})

    def compose(self, left, right, boundary, tokens, start=0, prefill=True, slot=0):
        ids = torch.tensor([tokens])
        positions = torch.arange(start, start+len(tokens))[None]
        slots = torch.tensor([slot])
        hidden = left.forward_layers(left.embedding(ids), positions, slots, prefill,
                                     start if prefill else 0, 0, boundary)
        hidden = right.forward_layers(hidden, positions, slots, prefill,
                                      start if prefill else 0, boundary, 4)
        return right.output_logits(hidden[:, -1])

    def test_every_boundary_matches_full_prefill_and_decode(self):
        for boundary in (1, 2, 3):
            with self.subTest(boundary=boundary):
                full, left, right = [self.fixture.executor() for _ in range(3)]
                prompt = [1, 3, 5, 7]
                torch.testing.assert_close(self.compose(left, right, boundary, prompt), full.prefill(prompt),
                                           rtol=0, atol=0)
                for start, token in enumerate((2, 4, 6), len(prompt)):
                    torch.testing.assert_close(self.compose(left, right, boundary, [token], start, False),
                                               full.decode(token)[0], rtol=0, atol=0)
                for name in ('kv', 'conv', 'state'):
                    for layer, tensor in getattr(full, name).items():
                        owner, unused = (left, right) if layer < boundary else (right, left)
                        torch.testing.assert_close(getattr(owner, name)[layer], tensor, rtol=0, atol=0)
                        self.assertEqual(torch.count_nonzero(getattr(unused, name)[layer]).item(), 0)

    def test_chunk_positions_and_interleaved_slot_are_preserved(self):
        full, left, right = [self.fixture.executor() for _ in range(3)]
        prompt = [1, 2, 3, 4, 5]
        self.compose(left, right, 2, prompt[:2])
        self.compose(left, right, 2, [7, 8], slot=1)
        full.prefill([7, 8], slot=1)
        torch.testing.assert_close(self.compose(left, right, 2, prompt[2:], 2), full.prefill(prompt),
                                   rtol=2e-5, atol=2e-6)
        for name in ('kv', 'conv', 'state'):
            for layer, tensor in getattr(full, name).items():
                owner = left if layer < 2 else right
                torch.testing.assert_close(getattr(owner, name)[layer], tensor, rtol=2e-5, atol=2e-6)

    def test_invalid_range_rejects_before_state_mutation(self):
        executor = self.fixture.executor()
        for start, end in ((-1, 2), (1, 1), (0, 5), (True, 3), (0, 1.5)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                executor.forward_layers(None, None, None, True, layer_start=start, layer_end=end)
        for tensors in (executor.kv, executor.conv, executor.state):
            self.assertTrue(all(torch.count_nonzero(t).item() == 0 for t in tensors.values()))
