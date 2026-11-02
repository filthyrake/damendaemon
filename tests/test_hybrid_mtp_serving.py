"""CPU property tests for MTP speculative serving (#68 decode lever).

Real CPU tensor arithmetic on the synthetic dense hybrid checkpoint (the
transcribed Qwen3.8-27B surface): packed optimistic spec rounds against
plain greedy references. No GPU execution or performance claim — the
exactness property is the test: every emission is the main model's greedy
argmax at its position, so the spec stream must equal plain greedy
token-for-token across accept, reject+corrective, packed-divergent-slot
and chunked-prefill paths.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from damendaemon.hybrid import HybridExecutor

from tests.test_hybrid_moe import (dense_root_config, dense_tensors,
                                   dense_mtp_tensors, LAYERS)
from damendaemon.config import decoder_config


class MtpServingTests(unittest.TestCase):
    def setUp(self):
        self.config = decoder_config(dense_root_config())

    def _model(self):
        state = dense_tensors(torch.Generator().manual_seed(123))
        state.update(dense_mtp_tensors(torch.Generator().manual_seed(456)))
        environment = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                       'DAMENDAEMON_EXPERIMENTAL_VISION': '0', 'DAMENDAEMON_HYBRID_MTP': '1'}
        with patch.dict('os.environ', environment, clear=False):
            from damendaemon.hybrid import load_weights
            weights = load_weights(state, self.config)
        weights = {name: value.float() for name, value in weights.items()}
        weights['head'] = weights['embed']
        return SimpleNamespace(config=self.config, max_context=64, weights=weights,
                                scales={}, quantization='none',
                                bytes=sum(t.numel()*t.element_size() for t in
                                          {id(t):t for t in weights.values()}.values()))

    def _executor(self):
        model = self._model()
        environment = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
                       'DAMENDAEMON_EXPERIMENTAL_VISION': '0', 'DAMENDAEMON_HYBRID_MTP': '1'}
        with patch.dict('os.environ', environment, clear=False), \
                patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.xpu_policy.validate_model', lambda *a, **k: None), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(model, use_graph=False, max_batch=2, kernels='torch')

    def _plain_greedy(self, executor, prompt, slot, count):
        logits = executor.prefill(prompt, slot)
        token = int(logits[0].float().argmax())
        output = [token]
        for _ in range(count - 1):
            logits, next_ids = executor.decode_batch([token], [slot])
            token = int(next_ids[0])
            output.append(token)
        return output

    def _spec_greedy(self, executor, prompt, slot, count, truth=None):
        """Accumulate a spec stream via packed spec rounds until `count`
        tokens; the final round may overshoot (the prefix is the claim).
        With `truth` (a known greedy reference), even rounds draft the
        true next token — forcing the accept path — while odd rounds seed
        from the draft head, exercising rejects and correctives."""
        logits = executor.prefill(prompt, slot)
        output = [int(logits[0].float().argmax())]
        rounds = accepted = 0
        while len(output) < count:
            second = (truth[len(output)] if truth is not None and rounds % 2 == 0
                      else None)
            blocks = [(slot, output[-1], second, False)]
            emissions, corrective, acc = executor.spec_round(blocks)
            accepted += acc
            rounds += 1
            for _, tokens in emissions:
                output.extend(tokens)
            if corrective:
                emissions, _, _ = executor.spec_round(corrective)
                for _, tokens in emissions:
                    output.extend(tokens)
        return output, rounds, accepted

    def test_spec_round_matches_plain_greedy(self):
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        count = 40
        plain = self._executor()
        expected = self._plain_greedy(plain, prompt, 0, count)
        spec = self._executor()
        # Interleave forced-true drafts with head seeds: accept, reject and
        # corrective paths must all keep the stream exactly greedy.
        output, rounds, accepted = self._spec_greedy(spec, prompt, 0, count, truth=expected)
        self.assertEqual(output[:count], expected)
        self.assertLessEqual(rounds, count - 1)
        self.assertGreater(accepted, 0, 'forced-true drafts must accept')

    def test_spec_round_accept_and_reject_paths(self):
        """Deterministic path coverage: a block whose draft IS the next
        reference token must accept and emit two tokens; a wrong draft
        must reject, emit the truth, and resolve its corrective block."""
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        plain = self._executor()
        expected = self._plain_greedy(plain, prompt, 0, 12)
        spec = self._executor()
        logits = spec.prefill(prompt, 0)
        self.assertEqual(int(logits[0].float().argmax()), expected[0])

        # Accept path: the reference's next token as the draft.
        emissions, corrective, accepted = spec.spec_round([(0, expected[0], expected[1], False)])
        self.assertEqual(accepted, 1)
        self.assertEqual(corrective, [])
        self.assertEqual(emissions[0][1], [expected[1], expected[2]])
        self.assertEqual(spec.lengths[0], len(prompt) + 2)

        # Reject path: a wrong draft (the truth is expected[3]).
        wrong = (expected[3] + 1) % self.config['vocab_size']
        if wrong == expected[3]:
            wrong = (wrong + 1) % self.config['vocab_size']
        emissions, corrective, accepted = spec.spec_round([(0, expected[2], wrong, False)])
        self.assertEqual(accepted, 0)
        self.assertEqual(emissions[0][1], [expected[3]])
        self.assertEqual(len(corrective), 1)
        slot, first, truth, flag = corrective[0]
        self.assertEqual((slot, first, truth, flag), (0, expected[2], expected[3], True))
        # The corrective block re-runs [first, truth] and emits the follower.
        emissions, _, _ = spec.spec_round(corrective)
        self.assertEqual(emissions[0][1], [expected[4]])
        self.assertEqual(spec.lengths[0], len(prompt) + 4)

    def test_spec_round_packed_divergent_slots(self):
        """Two slots at different lengths share one packed forward per
        round — the divergent per-row attention mask and per-slot state
        snapshots/rollback must keep each stream equal to its own plain
        greedy reference."""
        prompts = {0: [1, 2, 3, 7, 11, 5], 1: [4, 8, 15, 16, 23, 42, 9, 2, 6]}
        count = 30
        plain = self._executor()
        expected = {slot: self._plain_greedy(plain, prompt, slot, count)
                    for slot, prompt in prompts.items()}
        spec = self._executor()
        outputs = {}
        for slot, prompt in prompts.items():
            logits = spec.prefill(prompt, slot)
            outputs[slot] = [int(logits[0].float().argmax())]
            self.assertEqual(outputs[slot][0], expected[slot][0])
        rounds = 0
        while any(len(outputs[slot]) < count for slot in prompts):
            blocks = [(slot, outputs[slot][-1], None, False)
                      for slot in prompts if len(outputs[slot]) < count + 2]
            emissions, corrective, _ = spec.spec_round(blocks)
            rounds += 1
            for slot, tokens in emissions:
                outputs[slot].extend(tokens)
            if corrective:
                emissions, _, _ = spec.spec_round(corrective)
                for slot, tokens in emissions:
                    outputs[slot].extend(tokens)
        self.assertGreater(rounds, 1)
        for slot in prompts:
            self.assertEqual(outputs[slot][:count], expected[slot])

    def test_spec_round_after_chunked_prefill(self):
        """Chunked prefill coexistence: the incremental MTP KV fill during
        prefill_chunk must leave the draft head exactly as primed as the
        whole-prompt prefill (and the spec stream stays exact)."""
        prompt = list(range(2, 2 + 30))
        count = 24
        whole = self._executor()
        expected = self._plain_greedy(whole, prompt, 0, count)

        chunked = self._executor()
        pieces = [prompt[:7], prompt[7:19], prompt[19:]]
        logits = None
        with patch.object(type(chunked), 'supports_chunked_prefill',
                          new_callable=lambda: property(lambda self: True)):
            for index, piece in enumerate(pieces):
                logits = chunked.prefill_chunk(piece, 0, sum(map(len, pieces[:index])),
                                               final=index == len(pieces) - 1)
        self.assertEqual(int(logits[0].float().argmax()), expected[0])

        # The incremental MTP KV fill leaves the draft head exactly as
        # primed as the whole-prompt fill: identical entry pattern, pure
        # SDPA-route float noise (the masked-continuation vs is_causal
        # attention routes differ below argmax margins).
        self.assertTrue(torch.allclose(whole.kv[whole.mtp_layer],
                                        chunked.kv[chunked.mtp_layer], atol=1e-4),
                        'chunked MTP KV fill diverged structurally from the whole-prompt fill')

        output, _, _ = self._spec_greedy(chunked, prompt, 0, count)
        self.assertEqual(output[:count], expected)

    def test_spec_round_requires_head_and_prefill(self):
        spec = self._executor()
        spec.mtp_layer = None
        with self.assertRaisesRegex(ValueError, 'MTP draft head'):
            spec.spec_round([(0, 1, 2, False)])
        spec.mtp_layer = LAYERS
        # A block for a slot with no prefill (no stash) refuses loudly.
        with self.assertRaisesRegex(ValueError, 'prefill incomplete'):
            spec.spec_round([(1, 5, None, False)])


class MtpCausalPathTests(MtpServingTests):
    """#689: the MTP block's attention expresses the architectural hole as
    is_causal over keys sliced past position 0 (the fused path — no
    q-head-expanded K/V materialization) whenever rows are contiguous or
    single; the eager mask remains for divergent rows and as the
    DAMENDAEMON_MTP_EAGER_MASK parity control."""

    def test_env_validation(self):
        executor = self._executor()
        with patch.dict('os.environ', {'DAMENDAEMON_MTP_EAGER_MASK': 'yes'}, clear=False):
            with self.assertRaisesRegex(ValueError, 'MTP_EAGER_MASK'):
                executor.spec_decode([1, 2, 3, 7], 8)

    def test_sliced_causal_matches_eager_mask(self):
        """Whole-prompt MTP fill (contiguous multi-row) plus single-row
        proposals: the sliced-causal route and the legacy masked route
        must produce identical greedy streams."""
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        streams = {}
        for env in ('0', '1'):
            executor = self._executor()
            with patch.dict('os.environ', {'DAMENDAEMON_MTP_EAGER_MASK': env}, clear=False):
                output, _, _ = self._spec_greedy(executor, prompt, 0, 40)
            streams[env] = output
        self.assertEqual(streams['0'], streams['1'])

    def test_chunked_fill_parity_both_paths(self):
        """The incremental MTP KV fill during chunked prefill (fresh and
        continuation chunks) runs the same two routes; both must leave
        the KV structurally identical to the whole-prompt fill and the
        spec streams identical across routes."""
        prompt = list(range(2, 2 + 30))
        kvs, streams = {}, {}
        for env in ('0', '1'):
            executor = self._executor()
            mtp = executor.mtp_layer
            assert mtp is not None
            with patch.dict('os.environ', {'DAMENDAEMON_MTP_EAGER_MASK': env}, clear=False):
                with patch.object(type(executor), 'supports_chunked_prefill',
                                  new_callable=lambda: property(lambda self: True)):
                    for index, piece in enumerate([prompt[:7], prompt[7:19], prompt[19:]]):
                        logits = executor.prefill_chunk(piece, 0, sum(map(len, [prompt[:7], prompt[7:19], prompt[19:]][:index])),
                                                        final=index == 2)
            kvs[env] = executor.kv[mtp].clone()
            with patch.dict('os.environ', {'DAMENDAEMON_MTP_EAGER_MASK': env}, clear=False):
                output, _, _ = self._spec_greedy(executor, prompt, 0, 24)
            streams[env] = output
        self.assertTrue(torch.allclose(kvs['0'], kvs['1'], atol=1e-4),
                        'chunked MTP KV fill diverged between attention routes')
        self.assertEqual(streams['0'], streams['1'])

    def test_architectural_hole_never_read(self):
        """Position 0 has no MTP key: poisoning it with NaN must leave the
        spec stream untouched on the sliced-causal route (the fused path
        drops key 0 by slicing — a leaked read would surface as NaN logits
        and a diverged argmax)."""
        prompt = [1, 2, 3, 7, 11, 5, 42, 9]
        clean = self._executor()
        expected = self._plain_greedy(clean, prompt, 0, 16)
        poisoned = self._executor()
        layer = poisoned.mtp_layer
        assert layer is not None
        for part in poisoned.kv[layer]:   # k and v
            part[0, :, 0, :] = float('nan')
        with patch.dict('os.environ', {'DAMENDAEMON_MTP_EAGER_MASK': '0'}, clear=False):
            logits = poisoned.prefill(prompt, 0)
            self.assertTrue(torch.isfinite(logits[0].float()).all())
            output, _, _ = self._spec_greedy(poisoned, prompt, 0, 16)
        self.assertEqual(output[:16], expected)


if __name__ == '__main__':
    unittest.main()
