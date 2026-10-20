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
    runtime_options = {}
    pair_chunk_size = 32  # the fixture session context; production default is 128

    def extend_pair(self):
        """Subclass hook: extend the model table before the runtime claims."""

    @staticmethod
    def make_model():
        return SimpleNamespace(max_context=32, quantization='none', eos=frozenset(),
                               tokenizer=StubTokenizer(),
                               raw_tokens=lambda prompt: list(PROMPT),
                               chat_tokens=lambda messages, **options: list(PROMPT))

    def setUp(self):
        self.harness = ChainHarness(models=getattr(self, 'pair_models', 1))
        self.harness.start_workers()
        model = self.make_model()
        self.pair = ChainServingPair(self.harness.pipeline, model,
                                     identity=hashlib.sha256(b'chain-fixture').digest(),
                                     stage_layers=[[0, 1], [1, 3], [3, 4]],
                                     chunk_size=self.pair_chunk_size)
        self.extend_pair()
        self.runtime = ChainDistributedRuntime(self.pair, queue_timeout=5,
                                               **self.runtime_options)
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


class InterleavedChainServingTests(ChainServingTests):
    """Same fixtures, chunked: prompts flow through the scheduler's
    interleaved prefill path (PREFILL_CHUNK rounds) instead of whole-prompt
    PREFILL; output must be byte-equal to the whole-prompt reference."""
    runtime_options = {'interleaved_prefill': True}
    pair_chunk_size = 2

    def test_chunked_completion_matches_reference(self):
        self.start()
        base = self.serve()
        expected = self.executor_tokens(3)
        body = {'model': 'model', 'prompt': 'hello chain', 'max_tokens': 3}
        with self.post(base, body) as response:
            result = json.load(response)
        self.assertEqual(result['choices'][0]['text'],
                         ' '.join(f't{token}' for token in expected))
        self.assertEqual(result['usage']['completion_tokens'], 3)
        with urlopen(base + '/v1/models/status') as response:
            status = json.load(response)
        self.assertTrue(status['interleaved_prefill'])
        self.assertEqual(status['prefill_chunk_size'], 2)
        self.assertTrue(wait_until(lambda: not any(self.pair.state.lengths)))
        self.assertFalse(self.errors)

    def test_chunked_concurrent_requests_reuse_slots(self):
        self.start()
        base = self.serve()

        def request():
            with self.post(base, {'model': 'model', 'prompt': 'again',
                                  'max_tokens': 3}) as response:
                return json.load(response)
        results = [request() for _ in range(2)]
        expected = ' '.join(f't{token}' for token in self.executor_tokens(3))
        for result in results:
            self.assertEqual(result['choices'][0]['text'], expected)
        self.assertTrue(wait_until(lambda: self.runtime.completed == 2))
        self.assertTrue(wait_until(lambda: not any(self.pair.state.lengths)))


class SwitchingChainServingTests(ChainServingTests):
    """Two resident models over one chain: an alias switch swaps every rank's
    shard through the SWITCH round; a request for the other alias defers
    until the chain is idle, switches, and generates from the new weights."""

    pair_models = 2

    def extend_pair(self):
        # The alt model is structurally identical; identity differs. The
        # harness's 2-entry shard table backs the resident swap.
        self.pair.models['alt'] = self.make_model()
        self.pair.identities['alt'] = hashlib.sha256(b'chain-alt').digest()
        self.pair.aliases.append('alt')

    @staticmethod
    def alt_executor_tokens(count):
        """The ALT model's reference; same PROMPT tokens, different weights."""
        from tests.test_chain_pipeline import make_alt_model
        from tests.test_chain_pipeline import ChainHarness as _H
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            reference = HybridExecutor(make_alt_model(_H(1).model),
                                       use_graph=False, max_batch=2,
                                       kernels='torch')
            reference.prefill(PROMPT)
            tokens = []
            token = int(reference.decode(PROMPT[-1])[1][0])
            tokens.append(token)
            for _ in range(count - 1):
                token = int(reference.decode(token)[1][0])
                tokens.append(token)
        return tokens

    def test_switch_serves_alt_model_and_back(self):
        self.start()
        base = self.serve()
        # First request: primary model.
        body = {'model': 'model', 'prompt': 'hello chain', 'max_tokens': 2}
        with self.post(base, body) as response:
            first = json.load(response)
        primary_text = ' '.join(f't{t}' for t in super().executor_tokens(2))
        self.assertEqual(first['choices'][0]['text'], primary_text)
        self.assertTrue(wait_until(lambda: not any(self.pair.state.lengths)))
        # Second request: the ALT alias switches the resident shards.
        with self.post(base, dict(body, model='alt')) as response:
            second = json.load(response)
        alt_text = ' '.join(f't{t}' for t in self.alt_executor_tokens(2))
        self.assertEqual(second['choices'][0]['text'], alt_text)
        self.assertNotEqual(first['choices'][0]['text'], second['choices'][0]['text'])
        # Third request: back to the primary — the original weights replay.
        with self.post(base, body) as response:
            third = json.load(response)
        self.assertEqual(third['choices'][0]['text'], primary_text)
        with urlopen(base + '/v1/models/status') as response:
            status = json.load(response)
        self.assertEqual(status['model_switches'], 2)
        by_id = {entry['id']: entry for entry in status['models']}
        self.assertEqual(by_id['model']['residency'], 'gpu')
        self.assertEqual(by_id['alt']['residency'], 'host')
        self.assertEqual(by_id['alt']['model_sha256'],
                         hashlib.sha256(b'chain-alt').hexdigest())
        self.assertFalse(self.errors)

    def test_unknown_alias_rejects_without_switching(self):
        self.start()
        base = self.serve()
        before = self.pair.state.sequence
        with self.assertRaises(HTTPError) as error:
            self.post(base, {'model': 'missing', 'prompt': 'x', 'max_tokens': 1})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.pair.state.sequence, before)


if __name__ == '__main__':
    unittest.main()
