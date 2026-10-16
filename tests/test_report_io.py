"""Only bounded, readable JSON reports may replace prior evidence."""
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

    def test_unreadable_reports_never_replace_previous_evidence(self):
        from unittest.mock import patch
        from dllm.report_io import load_report
        invalid=(None,[],[{'passed':True}],True,1,'text',
                 {'metrics':{'1':0,1:1}}, {'metrics':{'null':0,None:1}},
                 {'text':'x'*256})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            for report in invalid:
                with self.subTest(report=report):
                    save_report(path,{'passed':False,'phase':'previous'})
                    previous=path.read_bytes()
                    with patch('dllm.report_io.MAX_REPORT_BYTES',128):
                        with self.assertRaises(ValueError):save_report(path,report)
                    self.assertEqual(path.read_bytes(),previous)
                    self.assertEqual(load_report(path),{'passed':False,'phase':'previous'})
                    self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_unambiguous_integer_keys_and_nested_arrays_remain_readable(self):
        from dllm.report_io import load_report
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            save_report(path,{'cases':{1:[None,True,{'value':3}]}})
            self.assertEqual(load_report(path),{'cases':{'1':[None,True,{'value':3}]}})

    def test_invalid_first_report_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            for report in ({'value':float('nan')},None,[],{'1':False,1:True}):
                with self.subTest(report=report),self.assertRaises(ValueError):save_report(path,report)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_finite_null_and_text_evidence_remains_compatible(self):
        report = {'missing': None, 'zero': 0, 'max': sys.float_info.max,
                  'tiny': 5e-324, 'text': 'NaN and Infinity are valid text', 'passed': False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            save_report(path, report)
            def reject(value): raise AssertionError('nonstandard number: '+value)
            self.assertEqual(json.loads(path.read_text(), parse_constant=reject), report)


class ReportReadTests(unittest.TestCase):
    def test_duplicate_fields_and_invalid_roots_are_rejected(self):
        from dllm.report_io import load_report
        values=('null','[]','{"passed":false,"passed":true}',
                '{"case":{"tokens":[1],"tokens":[2]}}','['*10000+']'*10000)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            for raw in values:
                with self.subTest(raw=raw[:70]):
                    path.write_text(raw)
                    with self.assertRaises(ValueError):load_report(path)

    def test_reader_bounds_actual_bytes_and_closes_on_failure(self):
        import io
        from unittest.mock import patch
        from dllm.report_io import load_report
        class Reader(io.BytesIO):
            def read(self,size=-1):
                self.requested=size
                return super().read(size)
        for size in (64,65):
            reader=Reader(b'{}'+b' '*(size-2))
            with patch('pathlib.Path.open',return_value=reader):
                if size==64:self.assertEqual(load_report('unused',max_bytes=64),{})
                else:
                    with self.assertRaises(ValueError):load_report('unused',max_bytes=64)
            self.assertEqual(reader.requested,65)
            self.assertTrue(reader.closed)

    def test_failed_numerical_diagnostics_remain_schema_owned(self):
        import math
        from dllm.report_io import load_report
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            path.write_text('{"passed":false,"error":NaN,"tokens":[1,2]}')
            report=load_report(path)
            self.assertFalse(report['passed'])
            self.assertTrue(math.isnan(report['error']))
            self.assertEqual(report['tokens'],[1,2])
