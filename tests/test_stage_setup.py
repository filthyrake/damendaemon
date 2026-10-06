import unittest
from unittest.mock import patch
import torch.distributed as dist
from dllm.stage_setup import StageSetup


class StageSetupTests(unittest.TestCase):
    def test_ready_handshake_and_distinct_sessions(self):
        store = dist.HashStore()
        first = StageSetup(dist.PrefixStore('first/', store), 0)
        peer = StageSetup(dist.PrefixStore('first/', store), 1)
        next_session = StageSetup(dist.PrefixStore('next/', store), 0)
        first.ready(); self.assertFalse(first.check_peer())
        peer.ready(); first.wait_ready(); peer.wait_ready()
        self.assertFalse(next_session.check_peer())

    def test_peer_failure_before_gpu_setup_is_visible(self):
        store = dist.HashStore(); first = StageSetup(store, 0); peer = StageSetup(store, 1)
        peer.fail(ValueError('private checkpoint path'))
        self.assertEqual(store.get('1'), b'failed:ValueError')
        with self.assertRaisesRegex(RuntimeError, 'peer stage preparation failed: ValueError'):
            first.ready()
        self.assertFalse(store.check(['0']))

    def test_missing_peer_and_malformed_status_are_bounded(self):
        store = dist.HashStore(); first = StageSetup(store, 0)
        first.ready()
        with patch('dllm.stage_setup.time.monotonic', return_value=first.deadline):
            with self.assertRaises(TimeoutError):first.wait_ready()
        store.set('1', b'garbage')
        with self.assertRaisesRegex(ValueError, 'invalid peer'):first.check_peer()

    def test_agent_store_is_required(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'torchrun agent store'):
                StageSetup.from_torchrun(0)
