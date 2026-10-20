"""Greedy streaming completion over the three-rank chain (CPU bus, real shards).

Mirrors the HostCompletion contract on the ChainHarness: token events with
finish_reason/usage, EOS and limit stops, slot release and reuse, single-owner
claim, cancellation between commands, and poisoned-epoch close semantics.
No serving or GPU claim.
"""
import threading
import unittest
from unittest.mock import patch

from damendaemon.chain_completion import ChainCompletion
from damendaemon.hybrid import HybridExecutor
from damendaemon.hybrid_pipeline import PipelineCancelled
from tests.test_chain_pipeline import ChainHarness, wait_until


class ChainCompletionHarness(ChainHarness):
    """Chain harness with an optional claimed completion owner."""

    def __init__(self, eos_ids=(), auto_claim=True):
        super().__init__()
        self.completion = (ChainCompletion(self.pipeline, eos_ids=eos_ids)
                           if auto_claim else None)


class TestChainCompletion(unittest.TestCase):
    def test_single_row_matches_executor_and_reports_usage(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            prompt = [3, 1, 4, 1, 5]
            with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                    patch('damendaemon.hybrid.empty_cache'), \
                    patch('damendaemon.hybrid.available_memory', return_value=2 ** 40):
                reference = HybridExecutor(harness.model, use_graph=False,
                                           max_batch=2, kernels='torch')
                reference.prefill(prompt)
                expected = []
                token = int(reference.decode(prompt[-1])[1][0])
                expected.append(token)
                for _ in range(3):
                    token = int(reference.decode(token)[1][0])
                    expected.append(token)
            events = list(harness.completion.generate(prompt, max_new_tokens=4))
            streamed = [event['token_id'] for event in events[:-1]]
            self.assertEqual(streamed, expected)
            final = events[-1]
            self.assertEqual(final['finish_reason'], 'length')
            self.assertEqual(final['usage'], dict(prompt_tokens=len(prompt),
                                                  completion_tokens=4,
                                                  total_tokens=len(prompt) + 4))
        finally:
            harness.close()

    def test_batch_with_eos_and_length_stops_releases_slots(self):
        harness = ChainCompletionHarness(auto_claim=False)
        harness.start_workers()
        try:
            # Discover row 0's first greedy token via a probe round, then
            # claim with that token as EOS so the stop is deterministic.
            probe = harness.pipeline.prefill([5, 6, 7], slot=0)[0]
            harness.pipeline.release([0])
            harness.completion = ChainCompletion(harness.pipeline, eos_ids={probe})
            events = list(harness.completion.generate_batch(
                [[5, 6, 7], [9, 8]], max_new_tokens=[2, 4]))
            tokens = {row: [event['token_id'] for event in events
                            if event.get('row') == row and 'token_id' in event]
                      for row in (0, 1)}
            self.assertEqual(tokens[0], [probe])
            self.assertEqual(len(tokens[1]), 4)
            reasons = {event['row']: event['finish_reason'] for event in events
                       if 'finish_reason' in event}
            self.assertEqual(reasons, {0: 'stop', 1: 'length'})
            # Both slots released. Each rank commits after the MIN ack, so the
            # coordinator's return does not imply the workers' cursors moved
            # yet — assert the final effect.
            self.assertTrue(wait_until(
                lambda: all(state.lengths == [0, 0] for state in harness.states)))
        finally:
            harness.close()

    def test_slot_reuse_across_requests(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            first = list(harness.completion.generate([5, 6, 7], max_new_tokens=2))
            second = list(harness.completion.generate([5, 6, 7], max_new_tokens=2))
            self.assertEqual([event['token_id'] for event in first[:-1]],
                             [event['token_id'] for event in second[:-1]])
            # Deterministic replay also proves the released slot was cleared:
            # a dirty recurrent state would diverge on the second request.
        finally:
            harness.close()

    def test_claim_and_concurrency_guards(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            with self.assertRaises(ValueError):
                ChainCompletion(harness.pipeline)
            with self.assertRaises(RuntimeError):
                stream = harness.completion.generate_batch([[5, 6, 7]],
                                                           max_new_tokens=2)
                next(iter(stream))
                list(harness.completion.generate_batch([[9, 8]],
                                                       max_new_tokens=2))
        finally:
            harness.close()

    def test_cancellation_between_commands(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            calls = []

            def cancelled():
                calls.append(1)
                return len(calls) >= 3

            stream = harness.completion.generate_batch([[5, 6, 7], [9, 8]],
                                                       max_new_tokens=4,
                                                       cancelled=cancelled)
            with self.assertRaises(PipelineCancelled):
                list(stream)
            self.assertTrue(harness.pipeline.failed or harness.pipeline.closed)
            self.assertTrue(harness.join_workers(timeout=10))
        finally:
            harness.close()

    def test_worker_failure_poisons_stream_and_closes(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            original = harness.stages[2].output_logits

            def exploding(*args, **kwargs):
                raise RuntimeError('injected head failure')

            harness.stages[2].output_logits = exploding
            stream = harness.completion.generate([5, 6, 7], max_new_tokens=3)
            with self.assertRaises(RuntimeError):
                list(stream)
            harness.stages[2].output_logits = original
            self.assertTrue(harness.pipeline.failed)
            # Worker ranks poison after their own MIN-ack contribution; assert
            # the final effect rather than the coordinator's return timing.
            self.assertTrue(wait_until(
                lambda: all(state.failed for state in harness.states)))
            self.assertTrue(harness.join_workers(timeout=10))
        finally:
            harness.close()

    def test_validation(self):
        harness = ChainCompletionHarness()
        harness.start_workers()
        try:
            completion = harness.completion
            with self.assertRaises(ValueError):
                list(completion.generate([5, 6, 7], max_new_tokens=0))
            with self.assertRaises(ValueError):
                list(completion.generate([5, 6, 7], max_new_tokens=31))
            with self.assertRaises(ValueError):
                list(completion.generate_batch([[5], [6], [7]]))
            with self.assertRaises(ValueError):
                list(completion.generate_batch([[5, 6, 7]], max_new_tokens=[1, 2]))
            with self.assertRaises(ValueError):
                list(completion.generate_batch([[5, 6, 7]], cancelled='no'))
            with self.assertRaises(ValueError):
                ChainCompletion(harness.pipeline, eos_ids={999999})
            # Validation errors never issued a command: the session is healthy.
            events = list(completion.generate([5, 6, 7], max_new_tokens=1))
            self.assertEqual(len(events), 2)
        finally:
            harness.close()


if __name__ == '__main__':
    unittest.main()
