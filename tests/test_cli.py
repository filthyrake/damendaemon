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


    def test_repository_can_select_one_gguf_file(self):
        with patch('dllm.cli.download', return_value={'path':'/cache/model.gguf'}) as download:
            self.assertEqual(resolve_model('owner/model', 'revision', 'model.gguf'), Path('/cache/model.gguf'))
            download.assert_called_once_with('owner/model', 'revision', filename='model.gguf')

    def test_local_path_cannot_silently_ignore_filename(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            resolve_model(directory, filename='model.gguf')

    def test_serve_assigns_templates_to_named_models(self):
        from dllm.cli import main
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory)/'chat.jinja'
            template.write_text('custom {{ messages }}')
            args = ['dllm','serve','--model','a='+directory,'--model','b='+directory,
                    '--chat-template','b='+str(template)]
            with patch('sys.argv',args), patch('dllm.scheduler.BatchedRuntime') as runtime, patch('dllm.server.serve'):
                main()
            self.assertEqual(runtime.call_args.kwargs['chat_templates'], {'b':'custom {{ messages }}'})
            for suffix in (['--chat-template','missing='+str(template)], ['--chat-template','b='+str(template)]):
                with patch('sys.argv',args+suffix), patch('dllm.scheduler.BatchedRuntime') as runtime, patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
                runtime.assert_not_called()
