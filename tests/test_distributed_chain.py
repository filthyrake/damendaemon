"""CPU tests for the #536 gate-4 three-rank chain control channel.

A deterministic in-memory transport stands in for NCCL: one shared bus holds
broadcast/P2P payloads in slots with events, so the command/ack/forward/return
ordering contracts are exercised without CUDA. No serving or GPU claim.
"""
import threading
import unittest

import torch

from damendaemon.distributed_chain import ChainChannel, ChainRole, ChainTransport
from damendaemon.stage_protocol import HEADER_WORDS


class FakeStageState:
    """Minimal stand-in for stage_protocol.StageState's begin/commit surface."""

    def __init__(self, header_words=8):
        self.header = torch.zeros(header_words, dtype=torch.int64)
        self.pending = None
        self.committed = []
        self.fail_begin = False
        self.failed = False

    def begin(self, words):
        if self.fail_begin:
            raise ValueError('rejected by state')
        if self.pending is not None:
            raise RuntimeError('state already has pending work')
        self.pending = tuple(words or self.header.tolist())
        return self.pending

    def commit(self, command):
        self.committed.append(command)
        self.pending = None

    def poison(self):
        self.failed = True
        self.pending = None


class FakeBus:
    """Shared slot bus for broadcasts, P2P handoffs and MIN reductions."""

    def __init__(self, world_size):
        self.world_size = world_size
        self._cond = threading.Condition()
        self._broadcast_epoch = 0
        self._broadcast_queues = {}  # reader rank -> FIFO of (epoch, tensor)
        self._reduce_cond = threading.Condition()
        self._reduce_values = []
        self._reduce_epoch = 0
        self._reduce_result = None

    # -- broadcast ---------------------------------------------------------
    def deliver_broadcast(self, tensor, source):
        """Queue the payload for every reader rank; the source never reads."""
        with self._cond:
            self._broadcast_epoch += 1
            for reader in range(self.world_size):
                if reader != source:
                    self._broadcast_queues.setdefault(reader, []).append(
                        (self._broadcast_epoch, tensor.clone()))
            self._cond.notify_all()

    def take_broadcast(self, tensor, reader):
        """Wait for this reader's oldest undelivered broadcast (FIFO).

        A round issues several broadcasts back-to-back (hidden hops, tokens)
        with no reduction barrier between them, so each reader consumes its
        own per-epoch FIFO instead of a shared slot.
        """
        with self._cond:
            queue = self._broadcast_queues.setdefault(reader, [])
            self._cond.wait_for(lambda: bool(queue), timeout=30)
            _, data = queue.pop(0)
            tensor.copy_(data)

    # -- min reduction -----------------------------------------------------
    def contribute_min(self, value):
        """All ranks block until everyone contributed; all read the min.

        Epoch-counted like the broadcast: the event from a previous
        reduction must not release the next one early. A timeout raises:
        returning a stale min would silently corrupt the ack (a release
        would commit on some ranks but not others).
        """
        with self._reduce_cond:
            epoch = self._reduce_epoch
            self._reduce_values.append(int(value))
            if len(self._reduce_values) == self.world_size:
                self._reduce_result = min(self._reduce_values)
                self._reduce_values.clear()
                self._reduce_epoch += 1
                self._reduce_cond.notify_all()

            def done():
                return self._reduce_epoch > epoch

            if not self._reduce_cond.wait_for(done, timeout=30):
                raise RuntimeError('fake bus reduction timed out')
            return self._reduce_result


class FakeTransport(ChainTransport):
    """NCCL-shaped: the source delivers, every other rank receives into its own buffer."""

    def __init__(self, rank, bus):
        self.rank = rank
        self.bus = bus
        self.ack = torch.tensor([1], dtype=torch.int64)

    def broadcast(self, tensor, source):
        if self.rank == source:
            self.bus.deliver_broadcast(tensor, source)
        else:
            self.bus.take_broadcast(tensor, self.rank)

    def all_reduce_min(self, tensor):
        tensor.fill_(self.bus.contribute_min(int(tensor.item())))

    def synchronize(self):
        pass


def make_chain(world_size=3):
    bus = FakeBus(world_size)
    states = [FakeStageState() for _ in range(world_size)]
    channels = [ChainChannel(ChainRole(r, world_size), FakeTransport(r, bus), states[r])
                for r in range(world_size)]
    return bus, states, channels


class TestChainRole(unittest.TestCase):
    def test_geometry(self):
        role = ChainRole(1, 3)
        self.assertEqual(role.forward_peer(), 2)
        self.assertFalse(role.owns_head)
        self.assertTrue(ChainRole(2, 3).owns_head)
        self.assertTrue(ChainRole(0, 3).is_coordinator)
        self.assertIsNone(ChainRole(2, 3).next_rank)
        with self.assertRaises(ValueError):
            ChainRole(2, 2).forward_peer()
        with self.assertRaises(ValueError):
            ChainRole(3, 3)
        with self.assertRaises(ValueError):
            ChainRole(0, 1)
        with self.assertRaises(ValueError):
            ChainRole(0, 0)


class TestBegin(unittest.TestCase):
    def test_broadcast_reaches_all_and_commits_pending(self):
        _, states, channels = make_chain(3)
        words = [1, 5, 9] + [0] * (HEADER_WORDS - 3)
        worker_results = {}

        def worker(rank):
            worker_results[rank] = channels[rank].begin(None)

        threads = [threading.Thread(target=worker, args=(r,)) for r in (1, 2)]
        for thread in threads:
            thread.start()
        accepted = channels[0].begin(words)
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(accepted, tuple(words))
        for state in states:
            self.assertEqual(state.pending, tuple(words))
        self.assertEqual(sorted(worker_results), [1, 2])

    def test_worker_cannot_originate(self):
        _, _, channels = make_chain(3)
        with self.assertRaises(ValueError):
            channels[1].begin([0] * HEADER_WORDS)

    def test_rejection_surfaces_on_every_rank(self):
        _, states, channels = make_chain(3)
        states[2].fail_begin = True
        words = [2] + [0] * (HEADER_WORDS - 1)
        worker_errors = {}

        def worker(rank):
            try:
                channels[rank].begin(None)
            except ValueError as error:
                worker_errors[rank] = error

        threads = [threading.Thread(target=worker, args=(r,)) for r in (1, 2)]
        for thread in threads:
            thread.start()
        with self.assertRaises(ValueError):
            channels[0].begin(words)
        for thread in threads:
            thread.join(timeout=5)
        self.assertIn(2, worker_errors)
        # rank 1 also rejected (MIN ack 0): both workers surface the same error.
        self.assertIn(1, worker_errors)


class TestTransfer(unittest.TestCase):
    """The broadcast-chain exchanges: every rank issues the same hop order."""

    def test_hidden_exchange_reaches_every_rank(self):
        bus, _, channels = make_chain(3)
        hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        received = {}

        def other(rank):
            buffer_ = torch.zeros(2, 3, dtype=torch.float32)
            channels[rank].exchange_hidden(buffer_, 0)
            received[rank] = buffer_

        threads = [threading.Thread(target=other, args=(r,)) for r in (1, 2)]
        for thread in threads:
            thread.start()
        channels[0].exchange_hidden(hidden, 0)
        for thread in threads:
            thread.join(timeout=5)
        self.assertTrue(torch.equal(received[1], hidden))
        self.assertTrue(torch.equal(received[2], hidden))

    def test_token_exchange_is_head_only(self):
        _, _, channels = make_chain(3)
        tokens = torch.tensor([7, 8], dtype=torch.int64)
        with self.assertRaises(ValueError):
            channels[0].exchange_tokens(tokens, 0)  # not the head rank
        with self.assertRaises(ValueError):
            channels[2].exchange_tokens(tokens, 0)  # head cannot receive its own hop
        received = {}

        def other(rank):
            buffer_ = torch.zeros(2, dtype=torch.int64)
            channels[rank].exchange_tokens(buffer_, 2)
            received[rank] = buffer_

        threads = [threading.Thread(target=other, args=(r,)) for r in (0, 1)]
        for thread in threads:
            thread.start()
        channels[2].exchange_tokens(tokens, 2)
        for thread in threads:
            thread.join(timeout=5)
        self.assertTrue(torch.equal(received[0], tokens))
        self.assertTrue(torch.equal(received[1], tokens))


class TestFinish(unittest.TestCase):
    def test_all_success_commits_everywhere(self):
        _, states, channels = make_chain(3)
        for state in states:
            state.pending = ('cmd',)
        results = {}

        def worker(rank):
            try:
                channels[rank].finish(('cmd',))
                results[rank] = 'ok'
            except RuntimeError as error:
                results[rank] = repr(error)

        threads = [threading.Thread(target=worker, args=(r,)) for r in (1, 2)]
        for thread in threads:
            thread.start()
        channels[0].finish(('cmd',))  # raises through if the MIN ack fails
        results[0] = 'ok'
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(results, {0: 'ok', 1: 'ok', 2: 'ok'})
        for state in states:
            self.assertEqual(state.committed, [('cmd',)])
            self.assertIsNone(state.pending)

    def test_one_failure_blocks_all_commits(self):
        _, states, channels = make_chain(3)
        for state in states:
            state.pending = ('cmd',)
        outcomes = {}

        def worker(rank, success):
            try:
                channels[rank].finish(('cmd',), success=success)
                outcomes[rank] = 'ok'
            except RuntimeError:
                outcomes[rank] = 'raised'

        threads = [threading.Thread(target=worker, args=(1, False)),
                   threading.Thread(target=worker, args=(2, True))]
        for thread in threads:
            thread.start()
        with self.assertRaises(RuntimeError):
            channels[0].finish(('cmd',), success=True)
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(outcomes[1], 'raised')
        self.assertEqual(outcomes[2], 'raised')
        # Nothing committed anywhere: a failed stage must not advance cursors.
        for state in states:
            self.assertEqual(state.committed, [])


if __name__ == '__main__':
    unittest.main()
