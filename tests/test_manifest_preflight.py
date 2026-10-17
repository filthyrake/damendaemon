"""Manifest failures must stop before model resolution or runtime imports."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
from damendaemon.model_manifest import load


def deep_manifest(path):
    path.write_text('{"version":1,"models":'+'['*10000+'0'+']'*10000+'}')


class ManifestPreflightTests(unittest.TestCase):
    def test_deep_manifest_is_validation_error_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'models.json';deep_manifest(path)
            resolve=Mock()
            with self.assertRaisesRegex(ValueError,'nesting'):load(path,resolve)
            resolve.assert_not_called()

    def test_cli_reports_validation_error_without_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'models.json';deep_manifest(path)
            flags=['-S']+(['-O'] if sys.flags.optimize else [])
            result=subprocess.run([sys.executable,*flags,'-m','damendaemon','serve','--model-config',str(path)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('manifest nesting',result.stderr)
            self.assertNotIn('Traceback',result.stderr)
            self.assertEqual(result.stdout,'')
