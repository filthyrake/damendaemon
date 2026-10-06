import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts.distributed_job import run_job


@unittest.skipUnless(os.name == 'posix', 'POSIX job ownership')
class DistributedJobTests(unittest.TestCase):
    def test_outer_termination_cleans_owned_job_before_harness_exit(self):
        child = ('import fcntl, json, os, signal, sys, time; '
                 'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                 'lock=open(sys.argv[1], "w"); fcntl.flock(lock, fcntl.LOCK_EX); '
                 'json.dump(dict(group=os.getpgrp()), lock); lock.flush(); time.sleep(60)')
        harness = ('import sys; from scripts.distributed_job import run_job; '
                   'run_job([sys.executable, "-c", sys.argv[1], sys.argv[2]], timeout=60, termination_grace=.2)')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'child.lock'
            process = subprocess.Popen([sys.executable, '-c', harness, child, str(path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            group = None
            cleaned = False
            try:
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    try:
                        group = json.loads(path.read_text())['group']
                        break
                    except (FileNotFoundError, json.JSONDecodeError):time.sleep(.01)
                self.assertIsNotNone(group, 'child did not acquire its lock')
                self.assertNotEqual(group, os.getpgrp())
                process.terminate()
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM)
                with path.open() as lock:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                cleaned = True
            finally:
                if not cleaned and group is not None and group != os.getpgrp():
                    try:os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:pass
                if process.poll() is None:process.kill()
                process.communicate(timeout=5)

    def test_output_and_nonzero_exit_are_preserved(self):
        previous = signal.getsignal(signal.SIGTERM)
        command = [sys.executable, '-c', 'import sys; print("out"); print("err", file=sys.stderr); sys.exit(3)']
        result = run_job(command, timeout=5, check=False)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (3, 'out\n', 'err\n'))
        with self.assertRaises(subprocess.CalledProcessError) as raised:run_job(command, timeout=5)
        self.assertEqual(raised.exception.stdout, 'out\n')
        self.assertEqual(raised.exception.stderr, 'err\n')
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_timeout_reaps_group_even_when_child_ignores_term(self):
        # The child holds both the capture pipe and an exclusive file lock.
        # Reacquiring the lock after timeout proves it was terminated, without
        # confusing a briefly unreaped zombie with a running process.
        child = ('import fcntl, signal, sys, time; '
                 'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                 'lock=open(sys.argv[1], "w"); fcntl.flock(lock, fcntl.LOCK_EX); '
                 'print("locked", flush=True); time.sleep(60)')
        parent = ('import subprocess, sys, time; '
                  'subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]]); time.sleep(60)')
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory)/'child.lock'
            started = time.monotonic()
            with patch('scripts.distributed_job.os.killpg', wraps=os.killpg) as signals:
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    run_job([sys.executable, '-c', parent, child, str(lock_path)],
                            timeout=1, termination_grace=.2)
                self.assertEqual([call.args[1] for call in signals.call_args_list],
                                 [signal.SIGTERM, signal.SIGKILL])
                self.assertTrue(all(call.args[0] != os.getpgrp() for call in signals.call_args_list))
            self.assertIn('locked', raised.exception.stdout)
            self.assertLess(time.monotonic()-started, 5)
            with lock_path.open() as lock:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
