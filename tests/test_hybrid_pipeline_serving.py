"""Real CPU pipeline through serving; device APIs are fixtures, not GPU proof."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

from dllm.hybrid_pipeline_build import PipelineBuildError
from dllm.hybrid_pipeline_policy import budgets
from dllm.runtime import Runtime, require_gpu
from dllm.scheduler import BatchedRuntime
from dllm.server import handler_for
from tests import test_hybrid_pipeline_build


class PipelineServingTests(unittest.TestCase):
    def setUp(self):
        fixture = test_hybrid_pipeline_build.PipelineBuildTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.model = fixture.model
        self.model.eos = set()
        self.model.raw_tokens = lambda tokens: list(tokens)
        self.model.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: ' '.join(map(str, tokens)))
        env = patch.dict('os.environ', DLLM_HYBRID_PIPELINE_BUDGETS_MIB='1024,1024',
                         CUDA_VISIBLE_DEVICES='GPU-first,GPU-second')
        env.start()
        self.addCleanup(env.stop)
        for name, kwargs in {
            'dllm.models.config': {'return_value': self.model.config},
            'dllm.models.load': {'return_value': self.model},
            'dllm.models.weight_bytes': {'return_value': self.model.bytes},
            'torch.cuda.set_device': {},
        }.items():
            replacement = patch(name, **kwargs)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_visible_devices_and_invalid_policy_reject_before_loading(self):
        require_gpu()
        for env in ({'CUDA_VISIBLE_DEVICES':'GPU-first'},
                    {'DLLM_HYBRID_PIPELINE_BUDGETS_MIB':'0,1024'},
                    {'DLLM_HYBRID_PIPELINE_BUDGETS_MIB':'1.5,1024'},
                    {'DLLM_EXPERIMENTAL_HYBRID_PIPELINE':'bad'}):
            with patch.dict('os.environ', env), self.assertRaises(ValueError):
                Runtime({'a':'a'}, context=32, use_graph=False)
        with patch('dllm.runtime.require_gpu') as gpu, self.assertRaises(ValueError):
            Runtime({'a':'a'}, context=32, use_graph=True)
        gpu.assert_not_called()
        with patch.dict('os.environ', DLLM_DEVICE='mps'), \
                patch('dllm.runtime.require_gpu') as gpu, self.assertRaises(ValueError):
            Runtime({}, context=32, use_graph=False)
        gpu.assert_not_called()
        with patch.dict('os.environ', DLLM_EXPERIMENTAL_HYBRID_PIPELINE='0'), self.assertRaises(ValueError):
            budgets()

    def test_scheduler_handles_full_and_chunked_prompts_and_concurrent_decode(self):
        for chunk in (0, 2):
            with self.subTest(chunk=chunk):
                runtime = BatchedRuntime({'a':'a'}, context=32, batch_size=2,
                                         use_graph=False, prefill_chunk_size=chunk)
                try:
                    def run(prompt):
                        return list(runtime.generate('a', None, max_tokens=3, raw_prompt=prompt))
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        results = list(pool.map(run, ([1, 2, 3, 4], [5, 6])))
                    for result in results:
                        self.assertEqual(result[-1]['usage']['completion_tokens'], 3)
                        self.assertEqual(result[-1]['timing']['kernel_backend'], 'torch-hybrid-pipeline')
                    self.assertFalse(runtime.runtime.gpu['a'].failed)
                finally:
                    runtime.close()
                    runtime.runtime.manage('unload', 'a', release_host=True)

    def test_failed_build_is_retained_until_retry_cleanup_succeeds(self):
        runtime = Runtime({'a':'a'}, context=32, use_graph=False, batch_size=2)
        failure = PipelineBuildError('injected', [], [])
        failure.close = Mock(side_effect=RuntimeError('still busy'))
        with patch('dllm.runtime.make_executor', side_effect=failure), self.assertRaises(PipelineBuildError):
            with runtime.session('a'):
                pass
        self.assertEqual(runtime.failed_builds, [failure])
        self.assertFalse(runtime.gpu)
        with patch('dllm.runtime.make_executor') as construct, self.assertRaisesRegex(RuntimeError, 'still busy'):
            with runtime.session('a'):
                pass
        construct.assert_not_called()
        failure.close.side_effect = None
        with runtime.session('a') as (executor, tier, _):
            self.assertEqual(tier, 'host')
            self.assertEqual(executor.kernel_backend, 'torch-hybrid-pipeline')
        self.assertFalse(runtime.failed_builds)
        runtime.manage('unload', 'a', release_host=True)

    def test_hot_swap_releases_every_stage_and_reports_pipeline_status(self):
        runtime = Runtime({'a':'a', 'b':'b'}, context=32, use_graph=False, batch_size=2)
        with runtime.session('a') as (first, _, __):
            first.prefill([1, 2])
        with runtime.session('b'):
            pass
        self.assertTrue(first.closed)
        self.assertTrue(all(stage.closed and not stage.w for stage in first.stages))
        status = runtime.manage('status')
        self.assertEqual(len(status['models'][1]['pipeline']['stages']), 2)
        self.assertEqual(status['pending_pipeline_cleanup'], 0)
        with runtime.session('a') as (restored, tier, _):
            self.assertEqual(tier, 'host')
            self.assertIsNot(restored, first)
            self.assertEqual(restored.lengths, [0, 0])
        runtime.manage('unload', 'a', release_host=True)
        runtime.manage('unload', 'b', release_host=True)

    def test_loopback_completions_reach_actual_cpu_pipeline(self):
        runtime = BatchedRuntime({'a':'a'}, context=32, batch_size=2,
                                 use_graph=False, prefill_chunk_size=2)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(runtime))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for stream in (False, True):
                payload = dict(model='a', prompt=[1, 2, 3], max_tokens=3, stream=stream)
                if stream:
                    payload['stream_options'] = {'include_usage':True}
                request = Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                                  json.dumps(payload).encode(), {'Content-Type':'application/json'})
                with urlopen(request, timeout=5) as response:
                    body = response.read().decode()
                if stream:
                    self.assertIn('data: [DONE]', body)
                    records = [json.loads(line[6:]) for line in body.splitlines()
                               if line.startswith('data: {')]
                    self.assertTrue(any(r.get('usage', {}).get('completion_tokens') == 3 for r in records
                                        if r.get('usage')))
                else:
                    self.assertEqual(json.loads(body)['usage']['completion_tokens'], 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            runtime.close()
            runtime.runtime.manage('unload', 'a', release_host=True)
