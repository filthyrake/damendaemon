from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.experiments.homer_cooling import CoolingGuard


class CoolingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for index, driver in enumerate(('k10temp', 'amdgpu')):
            path = self.root/f'hwmon{index}'
            path.mkdir()
            (path/'name').write_text(driver)
            (path/'temp1_input').write_text('45000')
            (path/'temp1_crit').write_text('95000')
        self.ec = self.root/'ec'
        self.ec.mkdir()
        for name, rpm in (('fan1', 4300), ('fan2', 4400), ('fan3', 1850)):
            directory = self.ec/name
            directory.mkdir()
            for key, value in (('mode', 'fixed'), ('level', '5'), ('rpm', str(rpm))):
                (directory/key).write_text(value)
        self.report = {}
        self.guard = CoolingGuard(self.report, hwmon_root=self.root, ec_root=self.ec)

    def test_cool_start_and_lower_exported_limit(self):
        self.guard.check(startup=True)
        self.assertEqual(self.report['maximum_observed_celsius'], 45)
        (self.root/'hwmon1/temp1_crit').write_text('85000')
        (self.root/'hwmon1/temp1_input').write_text('75000')
        with self.assertRaisesRegex(RuntimeError, 'thermal limit'):
            self.guard.check()
        self.assertEqual(self.report['maximum_observed_celsius'], 75)

    def test_wrong_fan_state_low_rpm_and_missing_gpu_fail_closed(self):
        for name, field, value in (('fan1', 'mode', 'auto'), ('fan2', 'level', '4'), ('fan3', 'rpm', '600')):
            path = self.ec/name/field
            original = path.read_text()
            path.write_text(value)
            with self.assertRaisesRegex(RuntimeError, 'fan-inference'):
                self.guard.check()
            path.write_text(original)
        (self.root/'hwmon1/temp1_input').unlink()
        with self.assertRaisesRegex(RuntimeError, 'both CPU and GPU'):
            self.guard.check()

    def test_hot_start_invalid_sensor_and_read_failure(self):
        path = self.root/'hwmon0/temp1_input'
        path.write_text('71000')
        with self.assertRaisesRegex(RuntimeError, 'cooldown'):
            self.guard.check(startup=True)
        path.write_text('999999')
        with self.assertRaisesRegex(RuntimeError, 'invalid temperature'):
            self.guard.check()
        path.write_text('unreadable')
        with self.assertRaises(ValueError):
            self.guard.check()
        self.assertIsNotNone(self.report['error'])

    def test_watchdog_stops_owned_child_on_thermal_failure(self):
        import subprocess
        import sys
        from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        report = {}
        (self.root/'hwmon0/temp1_input').write_text('80000')
        guard = HostMemoryWatchdog(child.kill, report, read=self.guard.read_host_available).start()
        try:
            self.assertNotEqual(child.wait(timeout=5), 0)
            self.assertTrue(report['triggered'])
            self.assertIn('thermal limit', report['error'])
        finally:
            if child.poll() is None: child.kill(); child.wait()
            guard.close()


class GuardLauncherTests(unittest.TestCase):
    def test_preflight_failure_never_launches_work_and_retains_report(self):
        from scripts.experiments import guard_homer
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/'guard.json'
            with patch('sys.argv', ['guard', '--report', str(report), '--', 'unused']), \
                    patch.object(guard_homer.socket, 'gethostname', return_value='HOMER'), \
                    patch.object(guard_homer.signal, 'signal'), \
                    patch.object(guard_homer, 'host_available', return_value=100*1024**3), \
                    patch.object(guard_homer, 'CoolingGuard') as cooling, \
                    patch.object(guard_homer.subprocess, 'Popen') as launch:
                cooling.return_value.check.side_effect = RuntimeError('fans not maximum')
                with self.assertRaisesRegex(RuntimeError, 'fans not maximum'):
                    guard_homer.main()
                launch.assert_not_called()
            import json
            value = json.loads(report.read_text())
            self.assertEqual(value['phase'], 'failed')
            self.assertTrue(value['cleanup_complete'])

    def test_deadline_kills_owned_child_and_checkpoints(self):
        import json
        import sys
        from scripts.experiments import guard_homer
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/'guard.json'
            with patch('sys.argv', ['guard', '--report', str(report), '--timeout', '1', '--',
                                    sys.executable, '-c', 'import time; time.sleep(60)']), \
                    patch.object(guard_homer.socket, 'gethostname', return_value='HOMER'), \
                    patch.object(guard_homer.signal, 'signal'), \
                    patch.object(guard_homer, 'host_available', return_value=100*1024**3), \
                    patch.object(guard_homer, 'CoolingGuard') as cooling:
                cooling.return_value.read_host_available.return_value = 100*1024**3
                with self.assertRaises(SystemExit):
                    guard_homer.main()
            value = json.loads(report.read_text())
            self.assertTrue(value['timed_out'])
            self.assertTrue(value['cleanup_complete'])
            self.assertNotEqual(value['exit_code'], 0)
