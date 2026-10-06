import os
import signal
import subprocess
import sys
import threading
import unittest

from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog


class HostMemoryWatchdogTests(unittest.TestCase):
    def test_low_headroom_stops_only_owned_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        report = {}
        guard = HostMemoryWatchdog(lambda: os.killpg(process.pid, signal.SIGKILL),
                                   report, reserve_bytes=20, read=lambda: 19).start()
        try:
            self.assertEqual(process.wait(timeout=3), -signal.SIGKILL)
            self.assertTrue(report['triggered'])
            self.assertEqual(report['minimum_available_bytes'], 19)
        finally:
            guard.close()
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_unreadable_memory_fails_closed(self):
        expired = threading.Event()
        report = {}
        def unreadable():
            raise OSError('missing accounting')
        guard = HostMemoryWatchdog(expired.set, report, read=unreadable).start()
        try:
            self.assertTrue(expired.wait(1))
            self.assertTrue(report['triggered'])
            self.assertIn('missing accounting', report['error'])
        finally:
            guard.close()

    def test_healthy_guard_stops_without_expiring(self):
        expired, sampled = threading.Event(), threading.Event()
        report = {}
        def read():
            sampled.set()
            return 30
        guard = HostMemoryWatchdog(expired.set, report, reserve_bytes=20, read=read).start()
        self.assertTrue(sampled.wait(1))
        guard.close()
        self.assertFalse(expired.is_set())
        self.assertFalse(guard.thread.is_alive())
