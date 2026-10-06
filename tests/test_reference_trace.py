"""Interrupted reference publication must retain the previous usable snapshot."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.reference_trace import save_trace


class ReferenceTraceTests(unittest.TestCase):
    def test_partial_does_not_replace_completed_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'golden.pt'
            save_trace(path, {'context': 512}, [{'targets': [torch.ones(3)]}], complete=True)
            save_trace(path, {'context': 1024}, [], complete=False)
            final = torch.load(path, weights_only=True)
            partial = torch.load(path.with_name('golden.partial.pt'), weights_only=True)
            self.assertTrue(final['complete'])
            self.assertEqual(final['context'], 512)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['context'], 1024)

    def test_failed_save_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'golden.pt'
            save_trace(path, {'context': 512}, [], complete=False)
            def fail_after_write(value, output):
                output.write(b'incomplete replacement')
                raise OSError('simulated full disk')
            with patch('scripts.reference_trace.torch.save', side_effect=fail_after_write):
                with self.assertRaises(OSError):
                    save_trace(path, {'context': 1024}, [], complete=False)
            partial = path.with_name('golden.partial.pt')
            self.assertEqual(torch.load(partial, weights_only=True)['context'], 512)
            self.assertEqual(list(Path(directory).iterdir()), [partial])
