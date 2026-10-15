"""Measurement integrity only; these fixtures are not GPU performance proof."""
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('modern_http', Path(__file__).parents[1] / 'scripts/benchmark_modern_http.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
summary_spec = importlib.util.spec_from_file_location('modern_summary', Path(__file__).parents[1] / 'scripts/summarize_modern_abba.py')
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


class Response(io.BytesIO):
    headers = Message()
    headers['Content-Type'] = 'text/event-stream'


class BenchmarkIntegrity(unittest.TestCase):
    def run_response(self, output=128, prompt=2, done=True, empty=False, requested=128, first=None):
        chunks = [dict(choices=[dict(text='' if empty else 'hello')]),
                  dict(choices=[], usage=dict(prompt_tokens=prompt, completion_tokens=output))]
        stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
        if done:
            stream += 'data: [DONE]\n\n'
        with patch.object(benchmark, 'urlopen', return_value=Response(stream.encode())) as post:
            result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3], threading.Barrier(1),
                                       output_tokens=requested, first_output_event=first)
            body = json.loads(post.call_args.args[0].data)
            self.assertEqual(body['prompt'], [1, 2])
            self.assertEqual(body['logit_bias'], {'3': -100})
            self.assertEqual(body['max_tokens'], requested)
        return result

    def test_fixed_prompt_preparation_retains_exact_shapes_and_offsets(self):
        tokenizer = Mock()
        tokenizer.encode.return_value.ids = list(range(2099))
        prompts = benchmark.prepare_fixed_prompts(tokenizer)
        for length in (128, 512, 2048):
            self.assertEqual(len(prompts[str(length)]), 4)
            for index, ids in enumerate(prompts[str(length)]):
                self.assertEqual(ids, list(range(index*17, index*17+length)))
        self.assertFalse(tokenizer.encode.call_args.kwargs['add_special_tokens'])

    def test_fixed_prompt_preparation_rejects_short_or_invalid_tokens(self):
        tokenizer = Mock()
        for ids in ([1]*2098, None, 'x'*2099, [True]*2099, [-1]*2099, [1.5]*2099):
            tokenizer.encode.return_value.ids = ids
            with self.assertRaises(ValueError):
                benchmark.prepare_fixed_prompts(tokenizer)

    def test_short_prompt_corpus_is_rejected_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(model_revision='fixture', engine_revision='fixture',
                                               precision='fixture', hardware='fixture', eos_ids=[3])))
            args = ['benchmark', '--base-url', 'http://127.0.0.1:1', '--engine', 'dllm',
                    '--tokenizer', str(root/'tokenizer.json'), '--manifest', str(manifest),
                    '--phase', 'A1', '--output', str(root/'result.json')]
            with patch('sys.argv', args), patch.object(benchmark.Tokenizer, 'from_file') as tokenizer, \
                    patch.object(benchmark, 'urlopen') as network:
                tokenizer.return_value.encode.return_value.ids = [1]*2098
                with self.assertRaisesRegex(ValueError, 'too short'):
                    benchmark.main()
            network.assert_not_called()
            self.assertFalse((root/'result.json').exists())

    def test_complete_usage(self):
        self.assertTrue(self.run_response()['passed'])

    def test_short_generation_fails(self):
        self.assertFalse(self.run_response(output=127)['passed'])

    def test_changed_prompt_count_fails(self):
        self.assertFalse(self.run_response(prompt=3)['passed'])

    def test_truncated_stream_fails(self):
        self.assertFalse(self.run_response(done=False)['passed'])

    def test_unterminated_done_fails_and_retains_output(self):
        chunks = [dict(choices=[dict(text='hello')]),
                  dict(choices=[], usage=dict(prompt_tokens=2, completion_tokens=128))]
        prefix = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
        for ending in ('data: [DONE]', 'data: [DONE]\n'):
            with patch.object(benchmark, 'urlopen', return_value=Response((prefix + ending).encode())):
                result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3],
                                           threading.Barrier(1))
            self.assertFalse(result['passed'])
            self.assertEqual(result['text'], 'hello')
            self.assertIn('unterminated SSE event', result['error'])

    def test_empty_output_fails(self):
        self.assertFalse(self.run_response(empty=True)['passed'])

    def test_long_context_output_reservation_is_enforced(self):
        event = threading.Event()
        self.assertTrue(self.run_response(output=512, requested=512, first=event)['passed'])
        self.assertTrue(event.is_set())
        self.assertFalse(self.run_response(output=128, requested=512)['passed'])

    def test_failed_stream_releases_first_output_waiter(self):
        event = threading.Event()
        self.assertFalse(self.run_response(empty=True, first=event)['passed'])
        self.assertTrue(event.is_set())

    def test_exact_usage_rejects_noninteger_and_conflicting_totals(self):
        valid = dict(prompt_tokens=2, completion_tokens=128)
        for changed in (dict(prompt_tokens=2.0, completion_tokens=128),
                        dict(prompt_tokens=2, completion_tokens=128.0),
                        dict(prompt_tokens=True, completion_tokens=128),
                        dict(prompt_tokens=2, completion_tokens=127), {}, [], False):
            with self.subTest(usage=changed):
                chunks = [dict(choices=[dict(text='kept')]), dict(usage=valid), dict(usage=changed)]
                stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
                first = threading.Event()
                with patch.object(benchmark, 'urlopen', return_value=Response(stream.encode())):
                    result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3],
                                               threading.Barrier(1), first_output_event=first)
                self.assertFalse(result['passed'])
                self.assertEqual(result['usage'], valid)
                self.assertEqual(result['text'], 'kept')
                self.assertNotIn('post_first_output_tokens_per_second_proxy', result)
                self.assertTrue(first.is_set())

    def test_identical_usage_repeats_and_null_intermediate_are_accepted(self):
        valid = dict(prompt_tokens=2, completion_tokens=128)
        chunks = [dict(usage=None), dict(choices=[dict(text='hello')]), dict(usage=valid), dict(usage=valid)]
        stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
        with patch.object(benchmark, 'urlopen', return_value=Response(stream.encode())):
            result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3], threading.Barrier(1))
        self.assertTrue(result['passed'])
        self.assertTrue(result['start_barrier_passed'])

    def test_failed_phase_checkpoint_preserves_previous_on_write_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'phase.json'
            output.write_text('{"previous": true}\n')
            tokenizer_path = root/'tokenizer.json'
            tokenizer_path.write_text('{}')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(model_revision='fixture', engine_revision='fixture',
                                               precision='fixture', hardware='fixture', eos_ids=[3])))
            args = ['benchmark', '--base-url', 'http://127.0.0.1:1', '--engine', 'dllm',
                    '--tokenizer', str(tokenizer_path), '--manifest', str(manifest),
                    '--phase', 'A1', '--output', str(output)]
            def interrupted(report, stream, **kwargs):
                stream.write('partial')
                raise OSError('injected write failure')
            for fail_write in (True, False):
                dump = summary.json.dump
                with patch('sys.argv', args), patch.object(benchmark.Tokenizer, 'from_file') as tokenizer, \
                        patch.object(benchmark, 'urlopen', return_value=Response(b'{"context":4096}')), \
                        patch.object(benchmark, 'request', side_effect=RuntimeError('request failed')), \
                        patch('dllm.report_io.json.dump', side_effect=interrupted if fail_write else dump):
                    tokenizer.return_value.encode.return_value.ids = list(range(3000))
                    with self.assertRaises(OSError if fail_write else RuntimeError):
                        benchmark.main()
                report = json.loads(output.read_text())
                if fail_write:
                    self.assertEqual(report, {'previous': True})
                else:
                    self.assertFalse(report['passed'])
                    self.assertIn('request failed', report['error'])
                self.assertEqual(list(root.glob('phase.json.*.tmp')), [])

    def test_output_capture_is_bounded_before_retention(self):
        cases = [(['é', 'é'], 4, 2, True, 'éé'),
                 (['é', 'é'], 3, 2, False, 'é'),
                 (['a', 'b', 'c'], 10, 2, False, 'ab'),
                 (['', 'abcd'], 4, 1, True, 'abcd'),
                 (['abcde'], 4, 2, False, '')]
        for fragments, byte_limit, count_limit, passed, expected in cases:
            with self.subTest(fragments=fragments, bytes=byte_limit, count=count_limit):
                chunks = [dict(choices=[dict(text=value)]) for value in fragments]
                chunks.append(dict(usage=dict(prompt_tokens=2, completion_tokens=128)))
                stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
                response = Response(stream.encode())
                first = threading.Event()
                with patch.object(benchmark, 'MAX_CAPTURE_BYTES', byte_limit), \
                        patch.object(benchmark, 'MAX_CAPTURE_FRAGMENTS', count_limit), \
                        patch.object(benchmark, 'urlopen', return_value=response):
                    result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3],
                                               threading.Barrier(1), first_output_event=first)
                self.assertEqual(result['passed'], passed)
                self.assertEqual(result['text'], expected)
                self.assertEqual(result['captured_utf8_bytes'], len(expected.encode()))
                self.assertLessEqual(len(result['chunks']), count_limit)
                self.assertEqual(result['capture_limits'], dict(utf8_bytes=byte_limit, fragments=count_limit))
                self.assertTrue(response.closed)
                self.assertTrue(first.is_set())
                if not passed:
                    self.assertIn('capture limit', result['error'])
                    self.assertNotIn('post_first_output_tokens_per_second_proxy', result)

    def test_broken_start_barrier_retains_failure_and_releases_waiter(self):
        barrier = threading.Barrier(2)
        barrier.abort()
        first = threading.Event()
        with patch.object(benchmark, 'urlopen') as post:
            result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3], barrier,
                                       first_output_event=first)
        post.assert_not_called()
        self.assertFalse(result['passed'])
        self.assertFalse(result['start_barrier_passed'])
        self.assertIn('BrokenBarrierError', result['error'])
        self.assertIsNone(result['first_output_seconds'])
        self.assertEqual(result['chunks'], [])
        self.assertTrue(first.is_set())


class SummaryIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for phase in ('A1', 'B1', 'B2', 'A2'):
            engine = 'dllm' if phase.startswith('A') else 'llama.cpp'
            report = dict(passed=True, phase=phase, engine=engine, prompts={}, context=4096,
                          output_tokens=128, tokenizer_sha256='fixture', eos_policy='fixture',
                          manifest=dict(model_revision='fixture', hardware='fixture', eos_ids=[1],
                                        engine_revision=engine, precision={}), rounds=[])
            for length in (128, 512, 2048):
                for concurrency in (1, 4):
                    for _ in range(10):
                        report['rounds'].append(dict(warmup=False, prompt_tokens=length,
                            concurrency=concurrency, seconds=2, requests=[dict(passed=True,
                            text='same', seconds=2, first_output_seconds=1,
                            usage=dict(prompt_tokens=length, completion_tokens=128)) for _ in range(concurrency)]))
            (self.directory/f'{phase}.json').write_text(json.dumps(report))

    def mutate(self, phase, fn):
        path = self.directory/f'{phase}.json'
        report = json.loads(path.read_text())
        fn(report)
        path.write_text(json.dumps(report))

    def test_measurement_is_not_competitive_verdict(self):
        result = summary.summarize(self.directory)
        self.assertIsNone(result['competitive_pass'])
        self.assertEqual(len(result['rows']), 6)

    def candidate_reports(self):
        for phase in ('A1', 'B1', 'B2', 'A2'):
            def convert(report):
                report['engine'] = 'dllm'
                report['manifest'].update(engine_revision='same-revision',
                    precision={'projections': 'Q8_0'}, candidate_source_sha256='same-source',
                    variant='ordinary' if phase.startswith('A') else 'hip-q8-small-batch')
            self.mutate(phase, convert)

    def test_throughput_uses_declared_and_verified_output_count(self):
        for phase in ('A1', 'B1', 'B2', 'A2'):
            def change(report):
                report['output_tokens'] = 64
                for row in report['rounds']:
                    for request in row['requests']:
                        request['usage']['completion_tokens'] = 64
            self.mutate(phase, change)
        result = summary.summarize(self.directory)
        self.assertEqual(result['output_tokens'], 64)
        for row in result['rows']:
            for engine in row['engines'].values():
                self.assertEqual(engine['aggregate_output_tokens_per_second']['p50'], 32 * row['concurrency'])

    def test_invalid_evidence_is_rejected(self):
        path = self.directory/'B2.json'
        original = path.read_text()
        mutations = [lambda r: r.update(passed=1), lambda r: r.update(output_tokens=128.0),
                     lambda r: r['rounds'][0]['requests'][0].update(passed='yes'),
                     lambda r: r['rounds'][0]['requests'][0].update(usage=None),
                     lambda r: r['rounds'][0]['requests'][0]['usage'].update(completion_tokens=128.0),
                     lambda r: r['rounds'][0]['requests'][0]['usage'].update(prompt_tokens=127),
                     lambda r: r['rounds'][0]['requests'][0].update(first_output_seconds=3),
                     lambda r: r['rounds'][0]['requests'][0].update(seconds=3),
                     lambda r: r['rounds'][0]['requests'][0].update(text='')]
        for bad in (0, -1, True, float('nan'), float('inf')):
            mutations.extend([lambda r, bad=bad: r['rounds'][0].update(seconds=bad),
                              lambda r, bad=bad: r['rounds'][0]['requests'][0].update(first_output_seconds=bad)])
        for mutate in mutations:
            path.write_text(original)
            self.mutate('B2', mutate)
            with self.assertRaises(ValueError):
                summary.summarize(self.directory)

    def test_each_round_occupancy_is_checked_not_only_pooled_count(self):
        def shift(report):
            rows = report['rounds']
            rows[1]['requests'].extend(rows[0]['requests'])
            rows[0]['requests'] = []
        self.mutate('B1', shift)
        with self.assertRaisesRegex(ValueError, 'each measured round'):
            summary.summarize(self.directory)

    def test_summary_failed_serialization_preserves_previous_file(self):
        output = self.directory/'summary.json'
        output.write_text('{"previous": true}\n')
        def interrupted(report, stream, **kwargs):
            stream.write('{"partial":')
            raise OSError('injected write failure')
        with patch('sys.argv', ['summary', str(self.directory), '--output', str(output)]), \
                patch('dllm.report_io.json.dump', side_effect=interrupted):
            with self.assertRaises(OSError):
                summary.main()
        self.assertEqual(json.loads(output.read_text()), {'previous': True})
        self.assertEqual(list(self.directory.glob('summary.json.*.tmp')), [])
        with patch('sys.argv', ['summary', str(self.directory), '--output', str(output)]):
            summary.main()
        self.assertTrue(json.loads(output.read_text())['workload_passed'])

    def test_candidate_comparison(self):
        self.candidate_reports()
        result = summary.summarize(self.directory)
        self.assertEqual(result['comparator'], 'dllm-candidate')
        self.assertTrue(all(row['exact_cross_engine_text'] for row in result['rows']))

    def test_candidate_changed_precision_rejected(self):
        self.candidate_reports()
        for phase in ('B1', 'B2'):
            self.mutate(phase, lambda r: r['manifest'].update(precision={'projections': 'FP16'}))
        with self.assertRaises(ValueError):
            summary.summarize(self.directory)

    def test_candidate_wrong_variant_rejected(self):
        self.candidate_reports()
        self.mutate('A2', lambda r: r['manifest'].update(variant='hip-q8-small-batch'))
        with self.assertRaises(ValueError):
            summary.summarize(self.directory)

    def test_missing_round_rejected(self):
        self.mutate('B2', lambda r: r['rounds'].pop())
        with self.assertRaises(ValueError):
            summary.summarize(self.directory)

    def test_changed_input_rejected(self):
        self.mutate('B1', lambda r: r.update(prompts={'128': [[1]]}))
        with self.assertRaises(ValueError):
            summary.summarize(self.directory)

    def test_text_mismatch_retained(self):
        self.mutate('B2', lambda r: r['rounds'][0]['requests'][0].update(text='different'))
        result = summary.summarize(self.directory)
        self.assertFalse(result['rows'][0]['exact_cross_engine_text'])
        self.assertFalse(result['rows'][0]['engines']['llama.cpp']['stable_text_per_prompt'])


if __name__ == '__main__':
    unittest.main()
