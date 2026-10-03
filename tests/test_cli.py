from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm.cli import resolve_model


class CliTests(unittest.TestCase):
    def test_existing_directory_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory, patch("dllm.cli.download") as download:
            self.assertEqual(resolve_model(directory), Path(directory).resolve())
            download.assert_not_called()

    def test_repository_name_resolves_requested_revision(self):
        with patch("dllm.cli.download", return_value={"path": "/cached/snapshot"}) as download:
            self.assertEqual(resolve_model("Qwen/Qwen2.5-0.5B-Instruct", "revision"), Path("/cached/snapshot"))
            download.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct", "revision")

    def test_missing_absolute_path_is_not_a_repository(self):
        with patch("dllm.cli.download") as download, self.assertRaises(ValueError):
            resolve_model("/missing-dllm-test-directory")
        download.assert_not_called()
