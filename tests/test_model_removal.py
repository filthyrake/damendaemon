"""Alias removal preserves files and independent registry/cache owners."""
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.cli import main
from dllm.model_manifest import load
from dllm.runtime import Runtime
from dllm.scheduler import BatchedRuntime
from dllm.server import handler_for


class ModelRemovalTests(unittest.TestCase):
    def test_http_cli_remove_preserves_shared_path_and_exports_corrected_registry(self):
        with tempfile.TemporaryDirectory() as directory, patch('dllm.runtime.require_gpu'), \
                patch('dllm.runtime.synchronize'), patch('dllm.runtime.empty_cache'):
            root = Path(directory)
            (root/'tokenizer_config.json').write_text('{}')
            checkpoint = root/'weights.safetensors'
            checkpoint.write_bytes(b'preserved checkpoint fixture')
            runtime = BatchedRuntime({'keep': root, 'wrong': root}, quantizations={'wrong': 'q4_0'})
            inner = runtime.runtime
            host = SimpleNamespace(bytes=1)
            inner.host.update(keep=host, wrong=host)
            removed_executor, kept_executor = Mock(), Mock()
            inner.gpu.update(keep=kept_executor, wrong=removed_executor)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(runtime))
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with patch.object(inner, '_status', side_effect=lambda: {'ids': inner.model_ids()}):
                    output = io.StringIO()
                    with patch('sys.argv', ['dllm', 'remove', 'wrong', '--server', base]), patch('sys.stdout', output):
                        main()
                    self.assertEqual(json.loads(output.getvalue()), {'ids': ['keep']})
                    removed_executor.close.assert_called_once()
                    kept_executor.close.assert_not_called()
                    self.assertNotIn('wrong', inner.host)
                    self.assertNotIn('wrong', inner.gpu)
                    self.assertEqual(set(inner.quantizations), {'keep'})
                    self.assertEqual(set(inner.chat_templates), {'keep'})
                    self.assertIs(inner.host['keep'], host)
                    self.assertEqual(checkpoint.read_bytes(), b'preserved checkpoint fixture')
                    request = Request(base+'/v1/models/remove', data=b'{"model":"wrong"}',
                                      headers={'Content-Type': 'application/json'})
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request, timeout=1)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                    exported = root/'models.json'
                    with patch('sys.argv', ['dllm', 'models', '--server', base, '--save-config', str(exported)]), patch('sys.stdout'):
                        main()
                    self.assertEqual(set(load(exported, Mock())[0]), {'keep'})
                    with patch('dllm.models.config', return_value={'max_position_embeddings': 4096}), \
                            patch('dllm.models.weight_bytes', return_value=1), patch('tokenizers.Tokenizer.from_file'):
                        request = Request(base+'/v1/models/register', data=json.dumps({
                            'model': 'wrong', 'path': directory, 'quantization': 'q8_0'}).encode(),
                            headers={'Content-Type': 'application/json'})
                        with urlopen(request, timeout=1) as response:
                            self.assertEqual(response.status, 200)
                    self.assertEqual(inner.quantizations['wrong'], 'q8_0')
                    self.assertEqual(set(runtime.model_manifest()['models']), {'keep', 'wrong'})
                    self.assertEqual(checkpoint.read_bytes(), b'preserved checkpoint fixture')
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)
                runtime.close()

    def test_failed_eviction_preserves_registered_identity_and_host_owner(self):
        with patch('dllm.runtime.require_gpu'):
            runtime = Runtime({'a': Path('/snapshot')}, quantizations={'a': 'q4_0'})
        runtime.gpu['a'] = object()
        runtime.host['a'] = object()
        before = runtime.model_manifest()
        with patch.object(runtime, '_evict', side_effect=RuntimeError('failed release')), \
                self.assertRaisesRegex(RuntimeError, 'failed release'):
            runtime.manage('remove', model='a')
        self.assertEqual(runtime.model_manifest(), before)
        self.assertIn('a', runtime.host)
        self.assertIn('a', runtime.gpu)
        self.assertFalse(runtime.lock.locked())
