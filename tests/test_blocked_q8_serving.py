import os
import unittest
from unittest import mock

import torch

from damendaemon import rocm_policy, thermal_pacing


class BlockedAdmissionTest(unittest.TestCase):
    def test_admits_only_gfx1151_with_hip(self):
        self.assertTrue(rocm_policy.blocked_matvec_admitted('6.4', 'gfx1151', False))
        self.assertTrue(rocm_policy.blocked_matvec_admitted(True, 'gfx1151:sramecc+:xnack-', False))

    def test_rejects_other_architectures_and_no_hip(self):
        self.assertFalse(rocm_policy.blocked_matvec_admitted('6.4', 'gfx1100', False))
        self.assertFalse(rocm_policy.blocked_matvec_admitted('6.4', 'gfx942', False))
        self.assertFalse(rocm_policy.blocked_matvec_admitted(None, 'gfx1151', False))
        self.assertFalse(rocm_policy.blocked_matvec_admitted(False, 'gfx1151', False))

    def test_kill_switch_blocks_admission(self):
        self.assertFalse(rocm_policy.blocked_matvec_admitted('6.4', 'gfx1151', True))

    def test_prefer_reads_kill_switch_environment(self):
        class FakeProperties:
            gcnArchName = 'gfx1151'

        fake_torch = mock.Mock()
        fake_torch.version.hip = '6.4'
        fake_torch.cuda.get_device_properties.return_value = FakeProperties
        with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': '0'}):
            self.assertFalse(rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda'))
        with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': 'off'}):
            self.assertFalse(rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda'))
        with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': 'false'}):
            self.assertFalse(rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda'))
        with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': ''}):
            self.assertTrue(rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda'))
        for value in ('1', 'true', 'on'):
            with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': value}):
                self.assertTrue(rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda'))
        for value in ('yes', 'enabled', '2'):
            with mock.patch.dict(os.environ, {'DAMENDAEMON_Q8_BLOCKED_MATVEC': value}):
                with self.assertRaises(ValueError):
                    rocm_policy.prefer_q8_blocked_matvec(fake_torch, 'cuda')


class ThermalPacingEnvTest(unittest.TestCase):
    def test_unset_environment_activates_nothing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(thermal_pacing.activate_from_env(environ=os.environ))

    def test_invalid_value_raises(self):
        for value in ('warm', 'nan', '-3', '120'):
            with mock.patch.dict(os.environ, {'DAMENDAEMON_THERMAL_SOFT_STOP': value}):
                with self.assertRaises(ValueError):
                    thermal_pacing.activate_from_env(environ=os.environ)

    def test_valid_value_returns_pacer_and_deactivate(self):
        with mock.patch.dict(os.environ, {'DAMENDAEMON_THERMAL_SOFT_STOP': '60'}):
            pacing = thermal_pacing.activate_from_env(environ=os.environ)
        self.assertIsNotNone(pacing)
        pacer, deactivate = pacing
        try:
            self.assertEqual(pacer.report['soft_stop_celsius'], 60.0)
            self.assertEqual(pacer.report['gates'], 0)
            self.assertEqual(pacer.report['events'], [])
        finally:
            deactivate()
        self.assertIn('min_transfer_bytes', pacer.report)

    def test_experiment_module_reexports_runtime(self):
        from scripts.experiments import thermal_pacing as scripts_pacing
        self.assertIs(scripts_pacing.ThermalPacer, thermal_pacing.ThermalPacer)
        self.assertIs(scripts_pacing.activate, thermal_pacing.activate)


if __name__ == '__main__':
    unittest.main()
