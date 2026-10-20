"""HTTP serving over the three-rank chain (CPU bus, real fixture shards).

Drives the proven DistributedRuntime scheduler through ChainServingPair on
the ChainHarness: real HTTP requests stream through the full chain, slots
release and reuse, idle heartbeats keep the epoch in lockstep, worker
failures close the session truthfully, and status metadata is exact.
No serving-performance or GPU claim.
"""
import hashlib
import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request as HttpRequest, urlopen

from damendaemon.chain_serving import (ChainDistributedRuntime, ChainServingPair,
                                       handler_for_chain)
from damendaemon.hybrid import HybridExecutor
from damendaemon.http_auth import AuthPolicy
from damendaemon.server import BoundedHTTPServer
from tests.test_chain_pipeline import ChainHarness, wait_until

PROMPT = [1, 3, 5, 7]


class StubTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return ' '.join(f't{token}' for token in tokens)

    decode_generated = decode


class ChainServingTests(unittest.TestCase):
    def setUp(self):
        self.harness = ChainHarness()
        self.harness.start_workers()
        model = SimpleNamespace(max_context=32, quantization='none', eos=frozenset(),
                                tokenizer=StubTokenizer(),
                                raw_tokens=lambda prompt: list(PROMPT),
                                chat_tokens=lambda messages, **options: list(PROMPT))
        self.pair = ChainServingPair(self.harness.pipeline, model,
                                     identity=hashlib.sha256(b'chain-fixture').digest(),
                                     stage_layers=[[0, 1], [1, 3], [3, 4]])
        self.runtime = ChainDistributedRuntime(self.pair, queue_timeout=5)
        self.owner = None
        self.errors = []
        self.addCleanup(self.close)

    def close(self):
        self.runtime.close()
        if self.owner is not None:
            self.owner.join(5)
            self.assertFalse(self.owner.is_alive())
        self.harness.close()

    def start(self, **options):
        def run():
            try:
                self.runtime.run(**options)
            except BaseException as error:
                self.errors.append(error)
        self.owner = threading.Thread(target=run, daemon=True)
        self.owner.start()

    def serve(self):
        server = BoundedHTTPServer(('127.0.0.1', 0),
                                   handler_for_chain(self.runtime, AuthPolicy()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def shutdown():
            server.shutdown(); server.server_close(); thread.join(3)
        self.addCleanup(shutdown)
        return f'http://127.0.0.1:{server.server_port}'

    def post(self, base, payload, path='/v1/completions'):
        return urlopen(HttpRequest(base + path, data=json.dumps(payload).encode(),
                                   headers={'Content-Type': 'application/json'}), timeout=6)

    def executor_tokens(self, count):
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            reference = HybridExecutor(self.harness.model, use_graph=False,
                                       max_batch=2, kernels='torch')
            reference.prefill(PROMPT)
            tokens = []
            token = int(reference.decode(PROMPT[-1])[1][0])
            tokens.append(token)
            for _ in range(count - 1):
                token = int(reference.decode(token)[1][0])
                tokens.append(token)
        return tokens

    def test_http_streaming_matches_executor_with_truthful_metadata(self):
        self.start()
        base = self.serve()
        expected = self.executor_tokens(3)
        text = ' '.join(f't{token}' for token in expected)
        body = {'model': 'model', 'prompt': 'hello chain', 'max_tokens': 3}
        with self.post(base, body) as response:
            result = json.load(response)
        self.assertEqual(result['choices'][0]['text'], text)
        self.assertEqual(result['usage'], dict(prompt_tokens=len(PROMPT),
                                               completion_tokens=3,
                                               total_tokens=len(PROMPT) + 3))
        timing = result['damendaemon_timing']
        self.assertEqual(timing['source_tier'], 'gpu')
        self.assertEqual(timing['kernel_backend'], 'distributed-chain-broadcast')
        with self.post(base, dict(body, stream=True,
                                  stream_options={'include_usage': True})) as response:
            wire = response.read().decode()
        events = [json.loads(line[6:]) for line in wire.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertEqual(''.join(event['choices'][0].get('text', '')
                                 for event in events if event['choices']), text)
        self.assertIn('data: [DONE]', wire)
        with urlopen(base + '/v1/models/status') as response:
            status = json.load(response)
        self.assertEqual(status['device'], 'distributed-cuda-chain')
        self.assertEqual(status['stage_layers'], [[0, 1], [1, 3], [3, 4]])
        self.assertEqual(status['models'][0]['model_sha256'],
                         self.pair.identity.hex())
        self.assertEqual(status['completed'], 2)  # non-stream + stream requests
        self.assertTrue(wait_until(lambda: not any(self.pair.state.lengths)))
        self.assertFalse(self.errors)

    def test_concurrent_slots_release_and_reuse(self):
        self.start()
        base = self.serve()

        def request():
            with self.post(base, {'model': 'model', 'prompt': 'again',
                                  'max_tokens': 3}) as response:
                return json.load(response)
        results = [request() for _ in range(2)]
        for result in results:
            self.assertEqual(result['usage']['completion_tokens'], 3)
        self.assertTrue(wait_until(lambda: self.runtime.completed == 2))
        self.assertTrue(wait_until(lambda: not any(self.pair.state.lengths)))
        # A third request reuses the released slots and still matches.
        third = request()
        self.assertEqual(third['choices'][0]['text'],
                         ' '.join(f't{token}' for token in self.executor_tokens(3)))

    def test_idle_heartbeat_keeps_epoch_alive(self):
        self.start(heartbeat_interval=0.02)
        self.serve()
        deadline = time.monotonic() + 3
        while self.runtime.heartbeats < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(self.runtime.heartbeats, 2)
        self.assertFalse(self.pair.pipeline.failed)
        # Heartbeats advanced the shared sequence without touching cursors.
        self.assertTrue(all(state.lengths == [0, 0] for state in self.harness.states))

    def test_invalid_requests_reject_before_chain_commands(self):
        self.start()
        base = self.serve()
        before = self.pair.state.sequence
        invalid = [{'model': 'model', 'prompt': 'x', 'max_tokens': 0},
                   {'model': 'model', 'prompt': 'x', 'max_tokens': 32},
                   {'model': 'model', 'prompt': 'x', 'max_tokens': 1, 'temperature': 1},
                   {'model': 'unknown', 'prompt': 'x', 'max_tokens': 1}]
        for body in invalid:
            with self.assertRaises(HTTPError) as error:
                self.post(base, body)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.pair.state.sequence, before)

    def test_worker_failure_fails_requests_and_closes_health(self):
        self.start()
        base = self.serve()
        original = self.harness.stages[2].output_logits

        def exploding(*args, **kwargs):
            raise RuntimeError('injected head failure')

        self.harness.stages[2].output_logits = exploding
        with self.assertRaises(HTTPError) as error:
            self.post(base, {'model': 'model', 'prompt': 'boom', 'max_tokens': 2})
        self.assertEqual(error.exception.code, 500)
        self.harness.stages[2].output_logits = original
        self.assertTrue(wait_until(lambda: self.pair.pipeline.failed
                                   and self.pair.pipeline.closed))
        with self.assertRaises(HTTPError) as error:
            urlopen(base + '/health')
        self.assertEqual(error.exception.code, 503)
        self.assertTrue(wait_until(lambda: self.harness.join_workers(timeout=5)))

    def test_validation_requires_unclaimed_healthy_pipeline(self):
        with self.assertRaises(ValueError):
            ChainServingPair(self.pair.pipeline, SimpleNamespace(),
                             identity=b'0' * 32, stage_layers=[[0, 1]])
        runtime = self.runtime
        runtime.close()
        with self.assertRaises(ValueError):
            ChainDistributedRuntime(self.pair)
        self.harness.pipeline._completion_owner = None
        with self.assertRaises(ValueError):
            ChainServingPair(self.harness.pipeline, self.pair.model,
                             identity=b'0' * 31, stage_layers=[[0, 1], [1, 3], [3, 4]])
        with self.assertRaises(ValueError):
            ChainServingPair(self.harness.pipeline, self.pair.model,
                             identity=b'0' * 32,
                             stage_layers=[[0, 1], [2, 4]])  # non-contiguous
        self.harness.pipeline._completion_owner = self.pair  # claimed now
        try:
            with self.assertRaises(ValueError):
                ChainServingPair(self.harness.pipeline, self.pair.model,
                                 identity=b'0' * 32,
                                 stage_layers=[[0, 1], [1, 3], [3, 4]])
        finally:
            self.harness.pipeline._completion_owner = None


if __name__ == '__main__':
    unittest.main()
