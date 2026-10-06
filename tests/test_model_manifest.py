import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dllm.model_manifest import load


class ManifestTests(unittest.TestCase):
    def test_restart_keeps_relative_paths_policies_and_template_text(self):
        from dllm.cli import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            path.write_text(json.dumps({'version': 1, 'models': {
                'chat': {'path': 'snapshots/pinned', 'quantization': 'q4_0',
                         'chat_template': '{{ messages }}'},
                'other': {'path': '/models/other', 'quantization': 'q8_0'}}}))
            calls = []
            for default in ('none', 'q8_0'):
                with patch.dict(os.environ, DLLM_QUANTIZATION=default), \
                        patch('sys.argv', ['dllm', 'serve', '--model-config', str(path)]), \
                        patch('dllm.scheduler.BatchedRuntime') as runtime, \
                        patch('dllm.server.serve'), patch('dllm.cli.download') as download:
                    main()
                    calls.append(runtime.call_args)
                    download.assert_not_called()
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(calls[0].args[0]['chat'], (path.parent/'snapshots/pinned').resolve())
            self.assertEqual(calls[0].kwargs['quantizations'], {'chat': 'q4_0', 'other': 'q8_0'})
            self.assertEqual(calls[0].kwargs['chat_templates'], {'chat': '{{ messages }}'})

    def test_repository_pin_and_single_format_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            entry = {'repository': 'Qwen/model', 'revision': 'a'*40,
                     'filename': 'chosen.gguf', 'quantization': 'none'}
            path.write_text(json.dumps({'version': 1, 'models': {'chat': entry}}))
            resolve = Mock(return_value=Path('/cache/chosen.gguf'))
            models, _, _ = load(path, resolve)
            resolve.assert_called_once_with('Qwen/model', 'a'*40, 'chosen.gguf')
            self.assertEqual(models['chat'], Path('/cache/chosen.gguf'))

    def test_invalid_document_cannot_download_an_earlier_valid_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            valid = {'repository': 'Qwen/model', 'revision': 'a'*40, 'quantization': 'q4_0'}
            for bad in ({'path': '/model'}, {**valid, 'revision': 'main'},
                        {**valid, 'path': '/model'}, {**valid, 'unknown': True},
                        {**valid, 'quantization': 'int4'}, {**valid, 'chat_template': 7},
                        {'path': '/model', 'revision': 'a'*40, 'quantization': 'none'}):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps({'version': 1, 'models': {'valid': valid, 'bad': bad}}))
                    resolve = Mock()
                    with self.assertRaises(ValueError):
                        load(path, resolve)
                    resolve.assert_not_called()
            path.write_text('{"version":1,"version":1,"models":{}}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load(path, Mock())

    def test_manifest_and_cli_model_options_cannot_silently_override(self):
        from dllm.cli import main
        for extra in (['--model', 'chat=/model'], ['--chat-template', 'chat=/template']):
            with patch('sys.argv', ['dllm', 'serve', '--model-config', '/missing']+extra), \
                    patch('dllm.model_manifest.load') as read, patch('sys.stderr'), self.assertRaises(SystemExit):
                main()
            read.assert_not_called()

    def test_runtime_uses_explicit_policies_and_rejects_invalid_before_gpu(self):
        from dllm.runtime import Runtime
        with patch('dllm.runtime.require_gpu') as gpu, patch.dict(os.environ, DLLM_QUANTIZATION='none'):
            runtime = Runtime({'a': Path('/model'), 'b': Path('/other')}, quantizations={'a': 'q4_0'})
            self.assertEqual(runtime.quantizations, {'a': 'q4_0', 'b': 'none'})
            gpu.reset_mock()
            for policies in ({'missing': 'q4_0'}, {'a': None}, {'a': 'int4'}):
                with self.assertRaises(ValueError):
                    Runtime({'a': Path('/model')}, quantizations=policies)
            gpu.assert_not_called()
