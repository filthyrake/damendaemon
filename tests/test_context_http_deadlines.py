"""Deadline/report integrity, not model or GPU performance qualification."""
import json
import io
from pathlib import Path
import tempfile
import threading
import signal
import subprocess
import sys
import select
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from scripts.verify_long_context_http import request_deadlines, collect_with_deadlines, save_report, check_overflow, stop_owned_server, expire_owned_request


class ContextDeadlineTests(unittest.TestCase):
    def test_overflow_requires_client_rejection_not_server_failure(self):
        for status, message, accepted in (
            (400, 'invalid or unsupported request', True),
            (400, 'unknown model', False),
            (503, 'invalid or unsupported request', False),
        ):
            with self.subTest(status=status, message=message):
                error = HTTPError('http://localhost', status, message, {},
                                  io.BytesIO(json.dumps({'error': message}).encode()))
                with patch('scripts.verify_long_context_http.urlopen', side_effect=error):
                    if accepted:
                        self.assertTrue(check_overflow('http://localhost', [1]*10)['passed'])
                    else:
                        with self.assertRaises(AssertionError):
                            check_overflow('http://localhost', [1]*10)

    def test_missing_first_output_expires(self):
        expired, first = threading.Event(), threading.Event()
        reasons = []
        def expire(reason):
            reasons.append(reason)
            expired.set()
        with request_deadlines(first, .01, 2, expire):
            self.assertTrue(expired.wait(1))
        self.assertEqual(reasons, ['first output'])

    def test_visible_output_does_not_disable_total_deadline(self):
        expired, first = threading.Event(), threading.Event()
        first.set()
        reasons = []
        def expire(reason):
            reasons.append(reason)
            expired.set()
        with request_deadlines(first, .01, .03, expire):
            self.assertTrue(expired.wait(1))
        self.assertEqual(reasons, ['total request'])

    def test_finished_request_cancels_both_deadlines(self):
        expired = threading.Event()
        with request_deadlines(threading.Event(), .05, .05, lambda _: expired.set()):
            pass
        self.assertFalse(expired.wait(.08))

    def test_expiry_overrides_successful_response_and_retains_evidence(self):
        for reason in ('first output', 'total request'):
            expired, first = threading.Event(), threading.Event()
            if reason == 'total request':
                first.set()
            original = dict(passed=True, text='retained', usage={'completion_tokens': 512},
                            post_first_output_tokens_per_second_proxy=50)
            def collect():
                self.assertTrue(expired.wait(1))
                return original
            result = collect_with_deadlines(collect, first, .01, .03, lambda _: expired.set())
            self.assertFalse(result['passed'])
            self.assertEqual(result['deadline_exceeded'], reason)
            self.assertEqual(result['text'], 'retained')
            self.assertEqual(result['usage'], original['usage'])
            self.assertNotIn('post_first_output_tokens_per_second_proxy', result)
            self.assertTrue(original['passed'])

    def test_first_expiry_is_recorded_once_and_callback_is_joined(self):
        class Timer:
            def __init__(self, seconds, callback):
                self.callback = callback
            def start(self):
                pass
            def cancel(self):
                pass
            def join(self):
                self.callback()  # Already-started callbacks can finish during join.
        reasons = []
        with patch('scripts.verify_long_context_http.threading.Timer', Timer):
            result = collect_with_deadlines(lambda: {'passed': True}, threading.Event(), 1, 2, reasons.append)
        self.assertEqual(reasons, ['first output'])
        self.assertEqual(result['deadline_exceeded'], 'first output')
        self.assertFalse(result['passed'])

    def test_in_budget_collection_is_unchanged(self):
        original = {'passed': True, 'text': 'normal'}
        expired = threading.Event()
        result = collect_with_deadlines(lambda: original, threading.Event(), 1, 2, lambda _: expired.set())
        self.assertIs(result, original)
        self.assertFalse(expired.is_set())

    def test_hard_deadline_stops_only_owned_child_ignoring_term(self):
        children = []
        try:
            for _ in range(2):
                child = subprocess.Popen([sys.executable, '-c',
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"],
                    start_new_session=True, stdout=subprocess.PIPE)
                children.append(child)
                self.assertTrue(select.select([child.stdout], [], [], 2)[0])
                self.assertEqual(child.stdout.readline(), b'ready\n')
            owned, unrelated = children
            stop_owned_server(owned, signal.SIGTERM)
            with self.assertRaises(subprocess.TimeoutExpired):
                owned.wait(timeout=.05)
            report = {}
            expire_owned_request(owned, report, 'total request')
            self.assertEqual(owned.wait(timeout=2), -signal.SIGKILL)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(report['deadline_exceeded'], 'total request')
            expire_owned_request(owned, report, 'first output')
            self.assertEqual(report['deadline_exceeded'], 'total request')
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)
                child.stdout.close()

    def test_owned_stop_handles_missing_or_already_exited_child(self):
        exited = Mock(pid=123, poll=Mock(return_value=0))
        with patch('scripts.verify_long_context_http.os.killpg') as kill:
            stop_owned_server(None, signal.SIGKILL)
            stop_owned_server(exited, signal.SIGKILL)
            kill.assert_not_called()
        running = Mock(pid=123, poll=Mock(return_value=None))
        with patch('scripts.verify_long_context_http.os.killpg', side_effect=ProcessLookupError) as kill:
            stop_owned_server(running, signal.SIGKILL)
            kill.assert_called_once_with(123, signal.SIGKILL)

    def test_checkpoint_remains_readable_and_replacement_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            save_report(path, {'passed': False, 'phase': 'long_request'})
            self.assertFalse(json.loads(path.read_text())['passed'])
            save_report(path, {'passed': False, 'interrupted_signal': 15})
            self.assertEqual(json.loads(path.read_text())['interrupted_signal'], 15)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_interrupted_write_cannot_corrupt_signal_handlers_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            dump = json.dump
            def interrupted(report, output, **kwargs):
                output.write('unfinished ordinary report')
                with patch('scripts.verify_long_context_http.json.dump', dump):
                    save_report(path, {'passed': False, 'interrupted_signal': 15})
                raise SystemExit(143)
            with patch('scripts.verify_long_context_http.json.dump', interrupted):
                with self.assertRaises(SystemExit):
                    save_report(path, {'phase': 'long_request'})
            self.assertEqual(json.loads(path.read_text()), {'passed': False, 'interrupted_signal': 15})
            self.assertEqual(list(Path(directory).iterdir()), [path])
