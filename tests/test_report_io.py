"""Nonfinite measurements must not replace prior JSON evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dllm.report_io import save_report


class ReportIOTests(unittest.TestCase):
    def test_nonfinite_nested_values_preserve_previous_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            save_report(path, {'passed': False, 'phase': 'previous'})
            previous = path.read_bytes()
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        save_report(path, {'passed': True, 'rounds': [{'metrics': [value]}]})
                    self.assertEqual(path.read_bytes(), previous)
                    self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_first_report_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            with self.assertRaises(ValueError): save_report(path, {'value': float('nan')})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_finite_null_and_text_evidence_remains_compatible(self):
        report = {'missing': None, 'zero': 0, 'max': sys.float_info.max,
                  'tiny': 5e-324, 'text': 'NaN and Infinity are valid text', 'passed': False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            save_report(path, report)
            def reject(value): raise AssertionError('nonstandard number: '+value)
            self.assertEqual(json.loads(path.read_text(), parse_constant=reject), report)
