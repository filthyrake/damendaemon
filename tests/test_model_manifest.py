import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from dllm.model_manifest import load, save


class ManifestTests(unittest.TestCase):
    def test_http_registration_export_cli_save_and_reload(self):
        from dllm.cli import main
        from dllm.runtime import Runtime
        from dllm.server import handler_for
        with tempfile.TemporaryDirectory() as directory, patch('dllm.runtime.require_gpu'):
            path = Path(directory)
            (path/'tokenizer_config.json').write_text('{}')
            runtime = Runtime({})
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(runtime))
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with patch('dllm.models.config', return_value={'max_position_embeddings': 4096}), \
                        patch('dllm.models.weight_bytes', return_value=100), \
                        patch('tokenizers.Tokenizer.from_file'), patch.object(runtime, '_status', return_value={}):
                    for alias, policy, template in (('chat', 'q4_0', '{{ messages }}'), ('plain', 'q8_0', None)):
                        body = dict(model=alias, path=directory, quantization=policy)
                        if template is not None:
                            body['chat_template'] = template
                        req = Request(base+'/v1/models/register', data=json.dumps(body).encode(),
                                      headers={'Content-Type': 'application/json'})
                        with urlopen(req) as response:
                            self.assertEqual(response.status, 200)
                output = path/'saved.json'
                with patch('sys.argv', ['dllm', 'models', '--server', base, '--save-config', str(output)]), patch('sys.stdout'):
                    main()
                resolve = Mock()
                models, policies, templates = load(output, resolve)
                resolve.assert_not_called()
                self.assertEqual(models, dict.fromkeys(('chat', 'plain'), path.resolve()))
                self.assertEqual(policies, {'chat': 'q4_0', 'plain': 'q8_0'})
                self.assertEqual(templates, {'chat': '{{ messages }}'})
                self.assertEqual(runtime.host, {})
                self.assertEqual(runtime.gpu, {})
                old = output.read_bytes()
                with runtime.registry_lock:
                    runtime.models.clear()
                with patch('sys.argv', ['dllm', 'models', '--server', base, '--save-config', str(output)]), self.assertRaises(HTTPError):
                    main()
                self.assertEqual(output.read_bytes(), old)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_failed_atomic_save_preserves_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            path.write_text('previous content')
            with patch('dllm.model_manifest.os.fsync', side_effect=OSError('disk failure')), self.assertRaises(OSError):
                save(path, {'version': 1, 'models': {}})
            self.assertEqual(path.read_text(), 'previous content')
            self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises(ValueError):
                save(path, {'oversized': 'a'*(1024*1024)})
            self.assertEqual(path.read_text(), 'previous content')

    def test_batched_manifest_export_does_not_use_gpu_queue(self):
        from dllm.scheduler import BatchedRuntime
        runtime = BatchedRuntime.__new__(BatchedRuntime)
        runtime.runtime = Mock()
        runtime._submit = Mock(side_effect=AssertionError('must not queue'))
        self.assertIs(runtime.model_manifest(), runtime.runtime.model_manifest.return_value)
        runtime._submit.assert_not_called()

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

    def test_invalid_later_gguf_selection_prevents_all_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            valid = {'repository': 'Qwen/model', 'revision': 'a'*40,
                     'quantization': 'none'}
            for filename in ('../model.gguf', '/model.gguf', '*.gguf',
                             'model.safetensors', 'model-00002-of-00002.gguf',
                             'model-00001-of-00065.gguf'):
                with self.subTest(filename=filename):
                    path.write_text(json.dumps({'version': 1, 'models': {
                        'first': valid, 'second': {**valid, 'filename': filename}}}))
                    resolve = Mock()
                    with self.assertRaises(ValueError):
                        load(path, resolve)
                    resolve.assert_not_called()

    def test_valid_split_selection_survives_manifest_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'models.json'
            filename = 'Q4/model-00001-of-00002.gguf'
            path.write_text(json.dumps({'version': 1, 'models': {'chat': {
                'repository': 'Qwen/model', 'revision': 'a'*40,
                'filename': filename, 'quantization': 'none'}}}))
            resolve = Mock(return_value=Path('/cache/model.gguf'))
            load(path, resolve)
            resolve.assert_called_once_with('Qwen/model', 'a'*40, filename)

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
