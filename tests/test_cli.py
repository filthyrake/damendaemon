from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from damendaemon.cli import resolve_model


class CliTests(unittest.TestCase):
    def test_existing_directory_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory, patch("damendaemon.cli.download") as download:
            self.assertEqual(resolve_model(directory), Path(directory).resolve())
            download.assert_not_called()

    def test_repository_name_resolves_requested_revision(self):
        with patch("damendaemon.cli.download", return_value={"path": "/cached/snapshot"}) as download:
            self.assertEqual(resolve_model("Qwen/Qwen2.5-0.5B-Instruct", "revision"), Path("/cached/snapshot"))
            download.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct", "revision")

    def test_missing_absolute_path_is_not_a_repository(self):
        with patch("damendaemon.cli.download") as download, self.assertRaises(ValueError):
            resolve_model("/missing-damendaemon-test-directory")
        download.assert_not_called()


    def test_repository_can_select_one_gguf_file(self):
        with patch('damendaemon.cli.download', return_value={'path':'/cache/model.gguf'}) as download:
            self.assertEqual(resolve_model('owner/model', 'revision', 'model.gguf'), Path('/cache/model.gguf'))
            download.assert_called_once_with('owner/model', 'revision', filename='model.gguf')

    def test_local_path_cannot_silently_ignore_filename(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            resolve_model(directory, filename='model.gguf')

    def test_serve_assigns_templates_to_named_models(self):
        from damendaemon.cli import main
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory)/'chat.jinja'
            template.write_text('custom {{ messages }}')
            args = ['damendaemon','serve','--model','a='+directory,'--model','b='+directory,
                    '--chat-template','b='+str(template)]
            with patch('sys.argv',args), patch('damendaemon.scheduler.BatchedRuntime') as runtime, patch('damendaemon.server.serve'):
                main()
            self.assertEqual(runtime.call_args.kwargs['chat_templates'], {'b':'custom {{ messages }}'})
            for suffix in (['--chat-template','missing='+str(template)], ['--chat-template','b='+str(template)]):
                with patch('sys.argv',args+suffix), patch('damendaemon.scheduler.BatchedRuntime') as runtime, patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
                runtime.assert_not_called()

    def test_serve_queue_limits_forward_and_invalid_values_fail_before_resolution(self):
        from damendaemon.cli import main
        args = ['damendaemon', 'serve', '--model', 'chat=/local/snapshot', '--queue-size', '12',
                '--queue-timeout', '.25']
        with patch('sys.argv', args), patch('damendaemon.scheduler.BatchedRuntime') as runtime, patch('damendaemon.server.serve'):
            main()
        self.assertEqual(runtime.call_args.kwargs['queue_size'], 12)
        self.assertEqual(runtime.call_args.kwargs['queue_timeout'], .25)
        for option, value in (('--queue-size', '0'), ('--queue-size', '2'),
                              ('--queue-size', '1.5'), ('--queue-timeout', '0'),
                              ('--queue-timeout', 'nan'), ('--queue-timeout', 'inf'),
                              ('--queue-timeout', '1e100')):
            with self.subTest(option=option, value=value), patch('sys.argv',
                    ['damendaemon', 'serve', '--model', 'chat=Qwen/model', option, value]), \
                    patch('damendaemon.cli.resolve_model') as resolve, patch('damendaemon.scheduler.BatchedRuntime') as runtime, \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                main()
            resolve.assert_not_called()
            runtime.assert_not_called()
