"""Measurement integrity only; these fixtures are not GPU performance proof."""
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import patch

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

    def test_complete_usage(self):
        self.assertTrue(self.run_response()['passed'])

    def test_short_generation_fails(self):
        self.assertFalse(self.run_response(output=127)['passed'])

    def test_changed_prompt_count_fails(self):
        self.assertFalse(self.run_response(prompt=3)['passed'])

    def test_truncated_stream_fails(self):
        self.assertFalse(self.run_response(done=False)['passed'])

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
                            text='same', seconds=2, first_output_seconds=1) for _ in range(concurrency)]))
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
