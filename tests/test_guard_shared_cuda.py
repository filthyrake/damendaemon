import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.guard_shared_cuda import Probe, check_service, run


class SharedCudaGuardTests(unittest.TestCase):
    def test_service_fails_closed(self):
        check_service({'status': 'ok'}, [{'is_processing': False}])
        for health, slots in [({}, []), ({'status': 'ok'}, []),
                              ({'status': 'ok'}, [{}]),
                              ({'status': 'ok'}, [{'is_processing': 0}]),
                              ({'status': 'ok'}, [{'is_processing': True}]),
                              ({'status': 'ok'}, [{'is_processing': False}, {}])]:
            with self.subTest(health=health, slots=slots), self.assertRaises(RuntimeError):
                check_service(health, slots)

    def test_busy_preflight_never_launches(self):
        report = {}
        with patch('scripts.experiments.guard_shared_cuda.subprocess.Popen') as launch:
            self.assertFalse(run([sys.executable, '-c', 'pass'],
                                 lambda **kw: check_service({'status': 'ok'}, [{'is_processing': True}]),
                                 1, report))
        launch.assert_not_called()
        self.assertFalse(report['child_started'])
        self.assertTrue(report['cleanup_complete'])

    def test_new_work_kills_only_owned_process(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder)/'started'
            incumbent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            calls = 0

            def probe(startup=False):
                nonlocal calls
                calls += 1
                if marker.exists():
                    raise RuntimeError('incumbent slot busy')
                return {}

            report = {}
            try:
                self.assertFalse(run([sys.executable, '-c',
                    'from pathlib import Path; import time; '
                    f'Path({str(marker)!r}).touch(); time.sleep(30)'], probe, 5, report))
                self.assertGreater(calls, 1)
                self.assertEqual(report['error'], 'incumbent slot busy')
                with self.assertRaises(ProcessLookupError):
                    os.kill(report['child_pid'], 0)
                self.assertIsNone(incumbent.poll())
            finally:
                incumbent.kill()
                incumbent.wait()

    def test_deadline_and_success(self):
        for code, timeout, expected in [('pass', 2, True), ('import time; time.sleep(30)', .1, False)]:
            report = {}
            self.assertEqual(run([sys.executable, '-c', code], lambda **kw: {}, timeout, report), expected)
            self.assertTrue(report['cleanup_complete'])
            if not expected:
                self.assertEqual(report['error'], 'experiment deadline')

    def test_sensor_identity_and_reserves(self):
        with patch('scripts.experiments.guard_shared_cuda.process_identity', return_value='123'), \
             patch('scripts.experiments.guard_shared_cuda.read_json', side_effect=lambda url:
                   {'status': 'ok'} if url.endswith('/health') else [{'is_processing': False}]), \
             patch('scripts.experiments.guard_shared_cuda.host_available', return_value=32*1024**3), \
             patch('scripts.experiments.guard_shared_cuda.subprocess.run') as telemetry:
            probe = Probe('GPU-example', 'http://localhost:8080', 1234)
            for output in ['GPU-example, 53, 4289', 'GPU-example, 65, 4289',
                           'GPU-example, 53, 4095', 'GPU-other, 53, 4289',
                           'GPU-example, N/A, 4289', '']:
                telemetry.return_value.stdout = output
                if output == 'GPU-example, 53, 4289':
                    self.assertEqual(probe(startup=True)['gpu_free_mib'], 4289)
                else:
                    with self.subTest(output=output), self.assertRaises((RuntimeError, ValueError, MemoryError)):
                        probe(startup=True)
            telemetry.return_value.stdout = 'GPU-example, 53, 2559'
            with self.assertRaises(MemoryError):
                probe()
            telemetry.return_value.stdout = 'GPU-example, 53, 4289'
            with patch('scripts.experiments.guard_shared_cuda.host_available', return_value=1):
                with self.assertRaises(MemoryError):
                    probe()
            with patch('scripts.experiments.guard_shared_cuda.process_identity', return_value='456'):
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    probe()


if __name__ == '__main__':
    unittest.main()
