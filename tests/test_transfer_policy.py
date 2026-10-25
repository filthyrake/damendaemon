"""#604 gate 3: transfer-policy selection, pinned staging bounds, peer disclosure."""
import sys
import types
import unittest
from unittest import mock

import torch

from damendaemon import transfer_policy
from damendaemon.transfer_policy import (PinnedStaging, enable_peer_access,
                                         peer_plan, pinned_staging_enabled)


class PinnedFlagTest(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict('os.environ', {}, clear=False):
            self.assertFalse(pinned_staging_enabled())

    def test_enable_and_disable_values(self):
        for value in ('1', 'true', 'on', ''):
            self.assertTrue(pinned_staging_enabled(value), value)
        for value in ('0', 'false', 'off'):
            self.assertFalse(pinned_staging_enabled(value), value)

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            pinned_staging_enabled('yes')

    def test_env_override(self):
        with mock.patch.dict('os.environ', {'DAMENDAEMON_PINNED_STAGING': '1'}):
            self.assertTrue(pinned_staging_enabled())


class PinnedStagingTest(unittest.TestCase):
    def test_disabled_matches_plain_copy(self):
        staging = PinnedStaging(torch.device('cpu'), max_transfer_bytes=2**20, enabled=False)
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = staging.to_device(value)
        self.assertTrue(torch.equal(result, value))  # plain .to(copy) passthrough
        back = staging.to_host(value)
        self.assertTrue(torch.equal(back, value))
        self.assertEqual(staging.bytes_staged, 0)

    def test_cpu_roundtrip_untouched_by_enabled_flag_on_cpu_device(self):
        staging = PinnedStaging(torch.device('cpu'), max_transfer_bytes=2**20, enabled=True)
        value = torch.randn(4, 4)
        self.assertTrue(torch.equal(staging.to_host(value), value))

    def test_budget_bounds_buffer_growth(self):
        staging = PinnedStaging(torch.device('cpu'), max_transfer_bytes=64, enabled=True)
        # A CPU-device staging degrades to plain .to; force the pinned branch
        # by faking a CUDA payload against a mocked device-side .to.
        payload = torch.ones(100, dtype=torch.float32)  # 400 bytes > 64-byte budget
        staged = PinnedStaging.__new__(PinnedStaging)
        staged.device = torch.device('cuda:0')
        staged.enabled = True
        staged.budget = 64
        staged._up = None
        staged._down = None
        staged.bytes_staged = 0
        with mock.patch.object(torch.Tensor, 'to', side_effect=RuntimeError('must not reach device')):
            with self.assertRaises(ValueError):
                staged.to_device(payload)

    def test_pinned_roundtrip_with_fake_cuda_payload(self):
        staged = PinnedStaging.__new__(PinnedStaging)
        staged.device = torch.device('cuda:0')
        staged.enabled = True
        staged.budget = 2**20
        staged._up = None
        staged._down = None
        staged.bytes_staged = 0
        payload = torch.randn(8, 8)

        def fake_pinned(numel, dtype):
            return torch.empty(numel, dtype=dtype)  # unpinned stand-in on CI

        with mock.patch.object(transfer_policy, '_pinned_empty', side_effect=fake_pinned), \
                mock.patch.object(transfer_policy, '_h2d',
                                  side_effect=lambda blob, device: blob):
            out = staged.to_device(payload)
        self.assertTrue(torch.equal(out, payload))
        self.assertEqual(staged.bytes_staged, 64 * 4)
        with mock.patch.object(transfer_policy, '_pinned_empty', side_effect=fake_pinned):
            back = staged.to_host(payload)
        self.assertTrue(torch.equal(back, payload))


class PeerPlanTest(unittest.TestCase):
    def test_empty_plan_without_cuda(self):
        torch_module = mock.Mock()
        torch_module.cuda.is_available.return_value = False
        self.assertEqual(peer_plan(torch_module), [])

    def test_pair_reported_with_error_recorded(self):
        torch_module = mock.Mock()
        torch_module.cuda.is_available.return_value = True
        torch_module.cuda.device_count.return_value = 2
        torch_module.cuda.can_device_access_peer.side_effect = RuntimeError('boom')
        plan = peer_plan(torch_module)
        self.assertEqual(len(plan), 2)
        self.assertFalse(plan[0]['peer_available'])
        self.assertIn('boom', plan[0]['error'])


class EnablePeerAccessTest(unittest.TestCase):
    def _torch_stub(self, cudart_result=0, mutual=True):
        torch_module = mock.Mock()
        torch_module.cuda.is_available.return_value = True
        torch_module.cuda.device_count.return_value = 2
        torch_module.cuda.can_device_access_peer.return_value = mutual
        cudart = mock.Mock()
        cudart.cudaDeviceEnablePeerAccess.return_value = cudart_result
        torch_module.cuda.cudart.return_value = cudart
        torch_module.cuda.device.return_value = mock.MagicMock()
        return torch_module, cudart

    def test_mutual_pair_enabled(self):
        torch_module, cudart = self._torch_stub(cudart_result=0)
        disclosures = enable_peer_access(torch_module)
        self.assertEqual(len(disclosures), 2)
        self.assertTrue(all(entry['enabled'] for entry in disclosures))
        self.assertEqual(cudart.cudaDeviceEnablePeerAccess.call_count, 2)

    def test_already_enabled_tolerated(self):
        torch_module, _ = self._torch_stub(cudart_result=701)
        disclosures = enable_peer_access(torch_module)
        self.assertTrue(all(entry['enabled'] for entry in disclosures))

    def test_runtime_error_disclosed_not_raised(self):
        torch_module, _ = self._torch_stub(cudart_result=2)
        disclosures = enable_peer_access(torch_module)
        self.assertTrue(all(not entry['enabled'] for entry in disclosures))
        self.assertIn('cudaDeviceEnablePeerAccess', disclosures[0]['error'])

    def test_non_mutual_pair_falls_back_with_disclosure(self):
        torch_module, cudart = self._torch_stub(mutual=False)
        disclosures = enable_peer_access(torch_module)
        self.assertTrue(all(not entry['enabled'] for entry in disclosures))
        self.assertEqual(cudart().cudaDeviceEnablePeerAccess.call_count, 0)


if __name__ == '__main__':
    unittest.main()
