"""Completion/failure ownership tests; real NCCL evidence is separate."""
import unittest
from unittest.mock import Mock, patch
from damendaemon.stage_protocol import Command, StageState, PREFILL
from damendaemon.stage_transport import StageChannel


class StageTransportTests(unittest.TestCase):
    def test_coordinator_validates_local_record_and_observes_peer_admission(self):
        channel = StageChannel.__new__(StageChannel)
        channel.state = StageState(1, 512)
        channel.rank = 0
        channel.header = Mock()
        channel.acknowledgement = Mock()
        channel.acknowledgement.item.return_value = 1
        command = Command(1, 1, PREFILL, 31, ((0, 1, 0),))
        with patch('damendaemon.stage_transport.torch.tensor'), patch('damendaemon.stage_transport.dist.broadcast'), \
                patch('damendaemon.stage_transport.dist.all_reduce') as admission:
            self.assertEqual(channel.begin(command), command)
            admission.assert_called_once()
        channel.header.cpu.assert_not_called()
        channel.acknowledgement.item.assert_called_once()
        self.assertEqual(channel.state.lengths, [0] * 4)

    def test_peer_rejection_poisoned_before_activation(self):
        for rank in (0, 1):
            channel = StageChannel.__new__(StageChannel)
            channel.state = StageState(1, 512)
            channel.rank = rank
            channel.header = Mock()
            command = Command(1, 1, PREFILL, 31, ((0, 1, 0),))
            channel.header.cpu.return_value.tolist.return_value = command.encode()
            channel.acknowledgement = Mock()
            channel.acknowledgement.item.return_value = 0
            with patch('damendaemon.stage_transport.torch.tensor'), patch('damendaemon.stage_transport.dist.broadcast'), \
                    patch('damendaemon.stage_transport.dist.all_reduce'):
                with self.assertRaisesRegex(ValueError, 'rejected before activation'):
                    channel.begin(command if rank == 0 else None)
            self.assertTrue(channel.state.failed)
            self.assertIsNone(channel.state.pending)
            self.assertEqual(channel.state.lengths, [0] * 4)
            self.assertEqual(channel.header.cpu.call_count, rank)

    def test_receiver_rejects_live_slot_before_activation_and_poisons(self):
        channel=StageChannel.__new__(StageChannel)
        channel.state=StageState(1,32)
        original=Command(1,1,PREFILL,4,((0,1,0),))
        channel.state.commit(channel.state.begin(original.encode()))
        channel.rank=1
        channel.header=Mock()
        channel.header.cpu.return_value.tolist.return_value=Command(1,2,PREFILL,3,((0,2,0),)).encode()
        channel.acknowledgement=Mock()
        channel.acknowledgement.item.return_value=0
        with patch('damendaemon.stage_transport.dist.broadcast'),patch('damendaemon.stage_transport.dist.all_reduce'):
            with self.assertRaisesRegex(ValueError,'rejected before activation') as error:channel.begin()
        self.assertIn('occupied slot',str(error.exception.__cause__))
        channel.acknowledgement.fill_.assert_called_once_with(0)
        self.assertTrue(channel.state.failed)
        self.assertIsNone(channel.state.pending)
        self.assertEqual(channel.state.generations,[1,0,0,0])
        self.assertEqual(channel.state.sequence,2)

    def pending(self):
        state = StageState(1, 512)
        command = state.begin(Command(1, 1, PREFILL, 31, ((0, 1, 0),)).encode())
        channel = StageChannel.__new__(StageChannel)
        channel.state = state
        channel.acknowledgement = Mock()
        return channel, command

    def test_commit_requires_host_observed_acknowledgement(self):
        channel, command = self.pending()
        def observe():
            self.assertEqual(channel.state.lengths[0], 0)
            self.assertIs(channel.state.pending, command)
            return 1
        channel.acknowledgement.item.side_effect = observe
        with patch('damendaemon.stage_transport.torch.cuda.synchronize'), patch('damendaemon.stage_transport.dist.all_reduce'):
            channel.finish(command)
        self.assertEqual(channel.state.lengths[0], 31)
        self.assertEqual(channel.state.sequence, 2)

    def test_failed_completion_fence_poisoned_before_publication(self):
        channel, command = self.pending()
        channel.acknowledgement.item.side_effect = RuntimeError('peer failed')
        with patch('damendaemon.stage_transport.torch.cuda.synchronize'), patch('damendaemon.stage_transport.dist.all_reduce'):
            with self.assertRaisesRegex(RuntimeError, 'peer failed'):channel.finish(command)
        self.assertTrue(channel.state.failed)
        self.assertEqual(channel.state.lengths, [0] * 4)
        self.assertEqual(channel.state.sequence, 1)
        self.assertIsNone(channel.state.pending)

    def test_invalid_token_completion_cannot_commit(self):
        import torch
        channel, command = self.pending()
        channel.rank = 0
        with self.assertRaisesRegex(ValueError, 'invalid stage token completion'):
            channel.complete_tokens(command, torch.zeros(1, dtype=torch.int64))
        self.assertTrue(channel.state.failed)
        self.assertEqual(channel.state.sequence, 1)
        self.assertEqual(channel.state.lengths, [0] * 4)

    def test_wrong_completion_rejected_before_collective(self):
        channel, command = self.pending()
        wrong = Command(1, 2, PREFILL, 31, ((0, 1, 0),))
        with patch('damendaemon.stage_transport.dist.all_reduce') as collective:
            with self.assertRaisesRegex(RuntimeError, 'does not own'):
                channel.finish(wrong)
            collective.assert_not_called()
        self.assertTrue(channel.state.failed)
        self.assertEqual(channel.state.sequence, 1)
