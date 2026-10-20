"""CPU tests for the #536 gate-4 chain pipeline coordinator (cycle 2).

Drives the real StageState command protocol and real HybridLayerStage shards
over the deterministic FakeBus transport: three worker threads (ranks 1-2)
against the coordinator (rank 0). Asserts FINAL effects — cursors commit on
all stages only after a fully successful round, and any stage failure poisons
every rank without advancing any cursor. No serving or GPU claim.
"""
import threading
import unittest
from unittest.mock import patch

import torch

from damendaemon.chain_pipeline import ChainPipeline, ChainStageWorker
from damendaemon.distributed_chain import ChainChannel, ChainRole
from damendaemon.hybrid import HybridExecutor
from damendaemon.hybrid_stage import HybridLayerStage
from damendaemon.stage_protocol import StageState
from tests import test_hybrid_layer_range
from tests.test_distributed_chain import FakeBus, FakeTransport


def make_harness():
    """Build the fixture chain; the fixture's env patch stays active for the
    harness lifetime (the eager-fp32 policy needs it) and is released in
    close() so it never leaks into other tests in the full suite."""
    setup = test_hybrid_layer_range.HybridLayerRangeTests()
    setup.setUp()
    return setup


def make_stage(model, start, end):
    with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
            patch('damendaemon.hybrid.empty_cache'), \
            patch('damendaemon.hybrid.available_memory', return_value=2**40):
        return HybridLayerStage(model, start, end, max_batch=2, kernels='torch')


class ChainHarness:
    """Three-rank chain over real StageState + fixture hybrid shards."""

    BOUNDS = (0, 1, 3, 4)

    def __init__(self):
        self._fixture = make_harness()
        self.model = self._fixture.fixture.model()
        self.bus = FakeBus(3)
        self.states = [StageState(epoch=1, context=32, slots=2) for _ in range(3)]
        self.channels = [ChainChannel(ChainRole(rank, 3), FakeTransport(rank, self.bus),
                                      self.states[rank]) for rank in range(3)]
        self.stages = [make_stage(self.model, self.BOUNDS[rank], self.BOUNDS[rank + 1])
                       for rank in range(3)]
        self.workers = {rank: ChainStageWorker(self.stages[rank], self.channels[rank])
                        for rank in (1, 2)}
        self.pipeline = ChainPipeline(self.stages[0], self.channels[0])
        self.threads = []

    def start_workers(self):
        for rank in (1, 2):
            thread = threading.Thread(target=self.workers[rank].run_forever)
            thread.start()
            self.threads.append(thread)

    def join_workers(self, timeout=10):
        for thread in self.threads:
            thread.join(timeout=timeout)
        return all(not thread.is_alive() for thread in self.threads)

    def close(self):
        joined = False
        try:
            if not self.pipeline.closed:
                try:
                    self.pipeline.close()
                except (ValueError, RuntimeError):
                    # A poisoned epoch cannot accept STOP; workers exit on
                    # their own once they see the poisoned state.
                    pass
            joined = self.join_workers()
        finally:
            self.bus = None
            self._fixture.doCleanups()
        return joined


def wait_until(predicate, timeout=5.0):
    """Wait for a worker thread's eventual commit; asserts FINAL effects."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class ChainPipelineTests(unittest.TestCase):
    def test_prefill_and_decode_match_full_model_and_commit_cursors(self):
        harness = ChainHarness()
        harness.start_workers()
        try:
            prompt = [3, 1, 4, 1, 5]
            with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                    patch('damendaemon.hybrid.empty_cache'), \
                    patch('damendaemon.hybrid.available_memory', return_value=2**40):
                reference = HybridExecutor(harness.model, use_graph=False, max_batch=2,
                                           kernels='torch')
                reference.prefill(prompt)
                token = int(reference.decode(prompt[-1])[1][0])
            first = harness.pipeline.prefill(prompt, slot=0)
            self.assertEqual(first, [token])
            chain_tokens = [token]
            for _ in range(4):
                with torch.inference_mode():
                    token = int(reference.decode(token)[1][0])
                expected = [token]
                self.assertEqual(harness.pipeline.decode_batch([chain_tokens[-1]], [0]),
                                 expected)
                chain_tokens.append(token)
            self.assertEqual(harness.pipeline.state.lengths[0], len(prompt) + 4)
            for state in harness.states[1:]:
                self.assertTrue(wait_until(lambda state=state:
                                           state.lengths[0] == harness.pipeline.state.lengths[0]))
                self.assertEqual(state.lengths[0], harness.pipeline.state.lengths[0])
                self.assertEqual(state.sequence, harness.pipeline.state.sequence)
        finally:
            self.assertTrue(harness.close())

    def test_decode_batch_divergent_slots(self):
        harness = ChainHarness()
        harness.start_workers()
        try:
            harness.pipeline.prefill([5, 6, 7], slot=0)
            harness.pipeline.prefill([9, 8], slot=1)
            self.assertEqual(harness.pipeline.state.lengths, [3, 2])
            first = harness.pipeline.decode_batch([6, 8], [0, 1])
            self.assertEqual(len(first), 2)
            self.assertEqual(harness.pipeline.state.lengths[0], 4)
            self.assertEqual(harness.pipeline.state.lengths[1], 3)
            for state in harness.states[1:]:
                self.assertTrue(wait_until(lambda state=state:
                                           state.lengths[:2] == harness.pipeline.state.lengths[:2]))
                self.assertEqual(state.lengths[:2], harness.pipeline.state.lengths[:2])
        finally:
            self.assertTrue(harness.close())

    def test_middle_stage_failure_poisons_without_cursor_advance(self):
        harness = ChainHarness()
        try:
            harness.start_workers()
            original = harness.stages[1].forward_layers

            def exploding(*args, **kwargs):
                raise RuntimeError('injected middle-stage failure')

            harness.stages[1].forward_layers = exploding
            with self.assertRaises(RuntimeError):
                harness.pipeline.prefill([1, 2, 3], slot=0)
            self.assertTrue(harness.join_workers(timeout=10))
            self.assertTrue(harness.pipeline.failed)
            for state in harness.states:
                self.assertTrue(state.failed)
                self.assertEqual(state.lengths, [0, 0])
                self.assertIsNone(state.pending)
            self.assertTrue(wait_until(lambda: all(worker.failed or worker.closed
                                                   for worker in harness.workers.values())))
            harness.stages[1].forward_layers = original
            with self.assertRaises(RuntimeError):
                harness.pipeline.prefill([1], slot=0)
        finally:
            harness.close()

    def test_head_stage_failure_returns_failed_token_and_poisons(self):
        harness = ChainHarness()
        try:
            harness.start_workers()
            original = harness.stages[2].output_logits

            def exploding(*args, **kwargs):
                raise RuntimeError('injected head-stage failure')

            harness.stages[2].output_logits = exploding
            with self.assertRaises(RuntimeError):
                harness.pipeline.prefill([2, 4, 6], slot=0)
            self.assertTrue(harness.join_workers(timeout=10))
            for state in harness.states:
                self.assertTrue(state.failed)
                self.assertEqual(state.lengths, [0, 0])
            harness.stages[2].output_logits = original
        finally:
            harness.close()

    def test_coordinator_stage_failure_poisons_everywhere(self):
        harness = ChainHarness()
        try:
            harness.start_workers()
            original = harness.stages[0].forward_layers

            def exploding(*args, **kwargs):
                raise RuntimeError('injected coordinator-stage failure')

            harness.stages[0].forward_layers = exploding
            with self.assertRaises(RuntimeError):
                harness.pipeline.prefill([7, 7], slot=0)
            self.assertTrue(harness.join_workers(timeout=10))
            for state in harness.states:
                self.assertTrue(state.failed)
                self.assertEqual(state.lengths, [0, 0])
            harness.stages[0].forward_layers = original
        finally:
            harness.close()

    def test_stop_closes_worker_loops(self):
        harness = ChainHarness()
        harness.start_workers()
        self.assertTrue(harness.close())
        for state in harness.states:
            self.assertTrue(state.closed)
        for worker in harness.workers.values():
            self.assertTrue(worker.closed)


if __name__ == '__main__':
    unittest.main()
