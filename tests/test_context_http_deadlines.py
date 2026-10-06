"""Deadline/report integrity, not model or GPU performance qualification."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.verify_long_context_http import request_deadlines, save_report


class ContextDeadlineTests(unittest.TestCase):
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
