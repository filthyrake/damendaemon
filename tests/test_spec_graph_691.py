"""CPU property tests for the #691 spec-round graph surface.

Real CPU tensor arithmetic on the synthetic dense hybrid MTP fixture:
the window-masked grouped attention (the graph body's read) must match
the eager per-row masked attention the spec round uses today, and the
static spec main forward (_spec_main_forward — the captured body) must
match the eager spec-round forward on identical state. No GPU execution
or performance claim: the exactness property is the test. Graph
capture/replay itself is the guarded A6000 cell's deliverable.
"""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from damendaemon.grouped_attention import window


def reference_window_attention(q, k, v, positions):
    """Eager per-row masked attention over the [:end] slice — the semantics
    the spec round's eager path implements (SDPA masked GQA)."""
    n, heads, length, dim = q.shape
    outputs = []
    for row in range(n):
        end = int(positions[row].max()) + 1
        mask = torch.arange(end).view(1, 1, 1, -1) <= positions[row].view(1, length, 1)
        out = F.scaled_dot_product_attention(q[row:row+1], k[row:row+1, :, :end],
                                             v[row:row+1, :, :end],
                                             attn_mask=mask, enable_gqa=True)
        outputs.append(out)
    return torch.cat(outputs, 0)


class GroupedWindowTests(unittest.TestCase):
    def test_window_matches_eager_masked_attention(self):
        torch.manual_seed(11)
        batch, kvh, heads, length, context, dim = 3, 2, 6, 2, 96, 8
        q = torch.randn(batch, heads, length, dim)
        k = torch.randn(batch, kvh, context, dim)
        v = torch.randn(batch, kvh, context, dim)
        # Divergent positions per batch row (the spec-round shape); tail
        # positions beyond each row's last are unwritten (zeros — the mask
        # must exclude them, mirroring unwritten KV cache positions).
        positions = torch.tensor([[5, 6], [40, 41], [90, 91]])
        k[0, :, 7:, :] = 0
        k[1, :, 42:, :] = 0
        got = window(q, k, v, positions)
        want = reference_window_attention(q, k, v, positions)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_window_rejects_bad_shapes_and_dtypes(self):
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 16, 8)
        positions = torch.zeros(1, 2, dtype=torch.long)
        window(q, k, v, positions)  # valid baseline
        with self.assertRaisesRegex(ValueError, 'matching FP32 GQA'):
            window(q, k, v, positions[:, :1])
        with self.assertRaisesRegex(ValueError, 'matching FP32 GQA'):
            window(q, k, v, positions.to(torch.int32))
        with self.assertRaisesRegex(ValueError, 'matching FP32 GQA'):
            window(q.half(), k.half(), v.half(), positions)
        with self.assertRaisesRegex(ValueError, 'matching FP32 GQA'):
            window(q[:1, :3], k, v, positions)  # heads not a GQA multiple

    def test_window_scratch_partitions_still_match(self):
        # A scratch budget forcing per-row partitions must not change the
        # result (the partition loop is behavior-preserving).
        torch.manual_seed(13)
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 2, 64, 8)
        v = torch.randn(1, 2, 64, 8)
        positions = torch.tensor([[30, 31]])
        whole = window(q, k, v, positions)
        tiny = window(q, k, v, positions, scratch_bytes=64*(2*1*2*4+1))
        torch.testing.assert_close(whole, tiny, rtol=0, atol=0)


class SpecMainForwardTests(unittest.TestCase):
    """The captured body (_spec_main_forward) vs the eager spec-round
    forward on identical executor state — the pre-capture parity contract
    (graph replay reproduces the captured math exactly, so eager parity of
    the body bounds replay parity)."""

    def _fixture(self):
        from tests.test_hybrid_mtp_serving import MtpServingTests
        host = MtpServingTests('test_spec_round_matches_plain_greedy')
        host.setUp()
        return host

    def test_spec_main_forward_matches_eager_forward(self):
        host = self._fixture()
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        # Twin executors from identical fixtures: body vs eager forward.
        body_executor = host._executor()
        eager_executor = host._executor()
        body_executor.prefill(prompt, 0)
        eager_executor.prefill(prompt, 0)
        # Any draft pair; only the main forward is compared.
        tokens = torch.tensor([[5, 6]])
        positions = torch.tensor([[len(prompt), len(prompt) + 1]])
        slots = torch.tensor([0])
        # Pre-fill KV identically is impossible via spec_round (it mutates);
        # compare the main forward only, which is the body's claim.
        body_hidden, body_logits = body_executor._spec_main_forward(tokens, positions, slots)
        eager_x = eager_executor.forward_layers(eager_executor.embedding(tokens),
                                                positions, slots, True, len(prompt))
        eager_logits = eager_executor.output_logits(eager_x)
        torch.testing.assert_close(body_logits, eager_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(body_hidden, eager_x, rtol=1e-4, atol=1e-4)
        # The recurrent state advances identically through both paths.
        for i in body_executor.state:
            torch.testing.assert_close(body_executor.state[i], eager_executor.state[i],
                                        rtol=1e-4, atol=1e-4)
        for i in body_executor.conv:
            torch.testing.assert_close(body_executor.conv[i], eager_executor.conv[i],
                                       rtol=1e-4, atol=1e-4)

    def test_spec_graph_env_validates_at_construction(self):
        # The opt-in env is parsed where the head loads — a garbage value
        # refuses executor construction (the CPU MTP fixture hits the
        # same block).
        host = self._fixture()
        with patch.dict('os.environ', {'DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH': 'x'},
                        clear=False):
            with self.assertRaisesRegex(ValueError, 'SPEC_GRAPH'):
                host._executor()


if __name__ == '__main__':
    unittest.main()
