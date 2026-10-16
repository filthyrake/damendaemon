from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import subprocess
import sys
import threading
import unittest
import time
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from dllm.bench import Result, events, request_one, summarize, llama_context, fixed_arrivals, distribution, evaluate_profile, profile_summary


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({'default_generation_settings': {'n_ctx': 1024}, 'total_slots': 4}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = request["model"]
        if model == "http-error":
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if model == 'sse-large':
            self.wfile.write(b'data: ' + b'x' * (1024 * 1024) + b'\n\n')
            return
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "several tokens in one chunk"}}]},
            {"choices": [{"delta": {"reasoning_content": "another output event"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        if model != "no-usage":
            chunks.append({"choices": [], "usage": {"completion_tokens": 7}})
        if model.startswith('text-field:'):
            _, field, encoded = model.split(':', 2)
            chunks[1]['choices'][0]['delta'] = {field: json.loads(encoded)}
        if model.startswith('envelope:'):
            _, field, encoded = model.split(':', 2)
            chunks[1][field] = json.loads(encoded)
        if model.startswith('tool-and-text:'):
            chunks[1]['choices'][0]['delta'] = {'content':'visible',
                'tool_calls':json.loads(model.split(':',1)[1])}
        if model == 'missing-finish':
            del chunks[3]
        elif model == 'invalid-finish':
            chunks[3]['choices'][0]['finish_reason'] = True
        elif model == 'late-output':
            chunks.append({'choices': [{'delta': {'content': 'too late'}}]})
        elif model == 'duplicate-finish':
            chunks.append(chunks[3])
        elif model == 'multiple-choices':
            chunks[3]['choices'].append({'delta': {}, 'finish_reason': 'stop'})
        elif model == 'invalid-delta':
            chunks[3]['choices'][0]['delta'] = []
        elif model.startswith('finish-'):
            chunks[3]['choices'][0]['finish_reason'] = model.removeprefix('finish-')
            chunks[3]['choices'][0]['delta'] = {'content': 'final output'}
        if model.startswith('usage-'):
            value = {'usage-over': {'completion_tokens': 9},
                     'usage-conflict': {'completion_tokens': 6},
                     'usage-repeat': {'completion_tokens': 7},
                     'usage-bool': {'completion_tokens': True},
                     'usage-negative': {'completion_tokens': -1},
                     'usage-float': {'completion_tokens': 7.0},
                     'usage-list': [], 'usage-false': False,
                     'usage-null': None, 'usage-empty': {}}[model]
            chunks.append({'choices': [], 'usage': value})
        if model.startswith('queue-'):
            values = {'queue-known': .125, 'queue-negative': -1, 'queue-nan': float('nan'),
                      'queue-inf': float('inf'), 'queue-bool': True, 'queue-huge': 10**400, 'queue-large': sys.float_info.max}
            chunks.append({'choices': [], 'dllm_timing': {'queue_seconds': values[model]}})
        try:
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if model == 'unterminated-done':
                self.wfile.write(b'data: [DONE]')
            elif model == 'single-newline-done':
                self.wfile.write(b'data: [DONE]\n')
            elif model != "truncated":
                self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # Invalid-stream tests deliberately close before the final event.


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, model):
        return request_one(self.url, model, "hello", 8, 5, 0)

    def test_comparator_context_is_verified_not_assumed(self):
        base = self.url.removesuffix('/v1')
        self.assertEqual(llama_context(base, 1024)['actual_context'], 1024)
        with self.assertRaisesRegex(RuntimeError, 'requested 2048, actual 1024'):
            llama_context(base, 2048)

    def test_duplicate_context_fields_cannot_hide_mismatches(self):
        for raw in (b'{"default_generation_settings":{"n_ctx":512,"n_ctx":1024}}',
                    b'{"default_generation_settings":{"n_ctx":512},"default_generation_settings":{"n_ctx":1024}}'):
            response=io.BytesIO(raw)
            with self.subTest(raw=raw),patch('dllm.bench.urlopen',return_value=response):
                with self.assertRaisesRegex(ValueError,'duplicate'):llama_context('http://local',1024)
            self.assertTrue(response.closed)

    def test_duplicate_stream_fields_cannot_hide_errors_or_usage(self):
        from types import SimpleNamespace
        for raw in ('{"error":"failed","error":null,"choices":[]}',
                    '{"choices":[],"usage":{"completion_tokens":9,"completion_tokens":7}}'):
            wire=('data: '+raw+'\n\n'+
                  'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'+
                  'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'+
                  'data: [DONE]\n\n').encode()
            response=io.BytesIO(wire)
            response.headers=SimpleNamespace(get_content_type=lambda:'text/event-stream')
            with self.subTest(raw=raw),patch('dllm.bench.urlopen',return_value=response):
                result=request_one('http://local','model','hello',8,5,0)
                self.assertEqual(result.error,'ValueError')
            self.assertTrue(response.closed)

    def test_response_nesting_is_a_closed_validation_failure(self):
        from types import SimpleNamespace
        raw=b'['*10000+b'0'+b']'*10000
        response=io.BytesIO(raw)
        with patch('dllm.bench.urlopen',return_value=response):
            with self.assertRaisesRegex(ValueError,'nesting'):llama_context('http://local',1024)
        self.assertTrue(response.closed)
        response=io.BytesIO(b'data: '+raw+b'\n\n')
        response.headers=SimpleNamespace(get_content_type=lambda:'text/event-stream')
        with patch('dllm.bench.urlopen',return_value=response):
            self.assertEqual(request_one('http://local','model','hello',8,5,0).error,'ValueError')
        self.assertTrue(response.closed)

    def test_context_metadata_read_is_bounded_and_closed(self):
        class TrackedResponse(io.BytesIO):
            sizes = None
            def read(self, size=-1):
                self.sizes.append(size)
                return super().read(size)
        prefix = b'{"default_generation_settings":{"n_ctx":1024}}'
        for padding, accepted in ((1024 * 1024 - len(prefix), True),
                                  (1024 * 1024 + 1 - len(prefix), False)):
            response = TrackedResponse(prefix + b' ' * padding)
            response.sizes = []
            with patch('dllm.bench.urlopen', return_value=response):
                if accepted:
                    self.assertEqual(llama_context('http://local', 1024)['actual_context'], 1024)
                else:
                    with self.assertRaisesRegex(ValueError, 'metadata.*limit'):
                        llama_context('http://local', 1024)
            self.assertEqual(response.sizes, [1024 * 1024 + 1])
            self.assertTrue(response.closed)

    def test_context_metadata_shapes_and_http_error_cleanup(self):
        from urllib.error import HTTPError
        for props in ([], None, {}, {'default_generation_settings': []},
                      {'default_generation_settings': None}):
            with self.subTest(props=props):
                response = io.BytesIO(json.dumps(props).encode())
                with patch('dllm.bench.urlopen', return_value=response):
                    with self.assertRaises(ValueError):
                        llama_context('http://local', 1024)
                self.assertTrue(response.closed)
        response = io.BytesIO(b'private error details')
        error = HTTPError('http://local/props', 503, 'unavailable', {}, response)
        with patch('dllm.bench.urlopen', side_effect=error):
            with self.assertRaises(HTTPError):
                llama_context('http://local', 1024)
        self.assertTrue(response.closed)

    def test_stream_usage_is_not_chunk_count(self):
        result = self.request("normal")
        self.assertIsNone(result.error)
        self.assertEqual(result.output_tokens, 7)
        self.assertEqual(result.output_events, 2)
        self.assertEqual(len(result.output_event_gaps_seconds), 1)
        self.assertGreaterEqual(result.output_event_gaps_seconds[0], 0)
        self.assertGreaterEqual(result.last_output_seconds, result.first_output_seconds)
        self.assertIsNotNone(result.first_output_seconds)
        self.assertGreaterEqual(result.seconds, result.first_output_seconds)

    def test_server_queue_timing_is_optional_and_validated(self):
        result = self.request('queue-known')
        self.assertIsNone(result.error)
        self.assertEqual(result.reported_server_queue_seconds, .125)
        self.assertEqual(summarize([result], 1)['reported_server_queue_seconds']['p99'], .125)
        result = self.request('normal')
        self.assertIsNone(result.reported_server_queue_seconds)
        self.assertIsNone(summarize([result], 1)['reported_server_queue_seconds'])
        for model in ('queue-negative', 'queue-nan', 'queue-inf', 'queue-bool'):
            with self.subTest(model=model):
                self.assertEqual(self.request(model).error, 'ValueError')

    def test_unrepresentable_server_queue_time_cannot_break_summary(self):
        results = [self.request('queue-huge') for _ in range(2)]
        # Two oversized integers previously overflowed statistics.median.
        summary = summarize(results, 1)
        self.assertEqual(summary['succeeded'], 0)
        self.assertEqual(summary['failed'], 2)
        self.assertIsNone(summary['reported_server_queue_seconds'])
        for result in results:
            self.assertEqual(result.error, 'ValueError')
            self.assertIsNone(result.reported_server_queue_seconds)

    def test_large_finite_queue_times_keep_a_finite_median(self):
        results = [self.request('queue-large') for _ in range(2)]
        self.assertTrue(all(result.error is None for result in results))
        summary = summarize(results, 1)
        self.assertEqual(summary['reported_server_queue_seconds']['p50'], sys.float_info.max)
        self.assertEqual(distribution([-sys.float_info.max] * 2)['p50'], -sys.float_info.max)
        self.assertEqual(distribution([-sys.float_info.max, sys.float_info.max])['p50'], 0)
        tiny = float.fromhex('0x0.0000000000001p-1022')
        self.assertEqual(distribution([tiny, tiny])['p50'], tiny)

    def test_invalid_or_conflicting_usage_cannot_inflate_goodput(self):
        profile = dict(max_first_output_seconds=5, max_output_event_gap_seconds=5,
                       min_request_output_tokens_per_second=.001)
        for model in ('usage-over', 'usage-conflict', 'usage-bool', 'usage-negative',
                      'usage-float', 'usage-list', 'usage-false'):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertEqual(result.error, 'ValueError')
                self.assertEqual(result.output_tokens, 7)  # Retain earlier partial evidence.
                self.assertEqual(summarize([result], 1)['succeeded'], 0)
                self.assertEqual(profile_summary([result], 1, profile)['qualifying_output_tokens'], 0)
        for model in ('usage-repeat', 'usage-null', 'usage-empty'):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertIsNone(result.error)
                self.assertEqual(result.output_tokens, 7)

    def test_malformed_text_deltas_cannot_count_as_output_or_goodput(self):
        profile = dict(max_first_output_seconds=5, max_output_event_gap_seconds=5,
                       min_request_output_tokens_per_second=.001)
        for field in ('content', 'reasoning_content', 'reasoning'):
            for value in (True, False, 1, 0, ['text'], [], {'text': 'hello'}, {}):
                with self.subTest(field=field, value=value):
                    result = self.request('text-field:' + field + ':' + json.dumps(value))
                    self.assertEqual(result.error, 'ValueError')
                    self.assertEqual(result.output_events, 0)
                    self.assertIsNone(result.first_output_seconds)
                    self.assertEqual(summarize([result], 1)['succeeded'], 0)
                    self.assertEqual(profile_summary([result], 1, profile)['qualifying_output_tokens'], 0)
        for field in ('content', 'reasoning_content', 'reasoning'):
            for value, count in ((None, 1), ('', 1), ('text', 2)):
                with self.subTest(field=field, value=value):
                    result = self.request('text-field:' + field + ':' + json.dumps(value))
                    self.assertIsNone(result.error)
                    self.assertEqual(result.output_events, count)

    def test_unknown_tokens_are_not_invented(self):
        result = self.request("no-usage")
        self.assertIsNone(result.error)
        self.assertIsNone(summarize([result], 1)["successful_output_tokens_per_second"])

    def test_truncated_stream_fails_even_with_usage(self):
        result = self.request("truncated")
        self.assertIsNotNone(result.error)
        self.assertEqual(summarize([result], 1)["succeeded"], 0)

    def test_unterminated_done_cannot_qualify(self):
        for model in ('unterminated-done', 'single-newline-done'):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertEqual(result.error, 'ValueError')
                self.assertEqual(result.output_tokens, 7)
                self.assertEqual(result.output_events, 2)
                self.assertEqual(summarize([result], 1)['succeeded'], 0)

    def test_done_cannot_substitute_for_valid_completion_termination(self):
        profile = dict(max_first_output_seconds=5, max_output_event_gap_seconds=5,
                       min_request_output_tokens_per_second=.001)
        for model in ('missing-finish', 'invalid-finish', 'late-output',
                      'duplicate-finish', 'multiple-choices', 'invalid-delta'):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertEqual(result.error, 'ValueError')
                self.assertGreater(result.output_events, 0)
                self.assertEqual(summarize([result], 1)['succeeded'], 0)
                self.assertEqual(profile_summary([result], 1, profile)['qualifying_output_tokens'], 0)

    def test_falsey_error_and_invalid_timing_envelopes_cannot_qualify(self):
        profile = dict(max_first_output_seconds=5, max_output_event_gap_seconds=5,
                       min_request_output_tokens_per_second=.001)
        for field, values in (('error', ({}, [], False, 0, '')),
                              ('dllm_timing', ([], False, 0, ''))):
            for value in values:
                with self.subTest(field=field,value=value):
                    result = self.request('envelope:' + field + ':' + json.dumps(value))
                    self.assertEqual(result.error, 'ValueError')
                    self.assertEqual(summarize([result],1)['succeeded'],0)
                    self.assertEqual(profile_summary([result],1,profile)['qualifying_output_tokens'],0)
        for field,value in (('error',None),('dllm_timing',None),('dllm_timing',{})):
            with self.subTest(field=field,value=value):
                result = self.request('envelope:' + field + ':' + json.dumps(value))
                self.assertIsNone(result.error)
                self.assertEqual(result.output_events,2)

    def test_malformed_tool_deltas_cannot_qualify(self):
        profile = dict(max_first_output_seconds=5, max_output_event_gap_seconds=5,
                       min_request_output_tokens_per_second=.001)
        invalid = [True, False, 1, 'call', {}, [None], [True], [{}],
                   [{'index':True}], [{'index':-1}], [{'index':0,'id':1}],
                   [{'index':0,'type':'custom'}], [{'index':0,'function':[]}],
                   [{'index':0,'function':{'name':True}}],
                   [{'index':0,'function':{'arguments':{}}}]]
        for calls in invalid:
            with self.subTest(calls=calls):
                result = self.request('text-field:tool_calls:' + json.dumps(calls))
                self.assertEqual(result.error, 'ValueError')
                self.assertEqual(result.output_events, 0)
                self.assertEqual(summarize([result], 1)['succeeded'], 0)
                self.assertEqual(profile_summary([result], 1, profile)['qualifying_output_tokens'], 0)

    def test_text_cannot_hide_invalid_tool_fragment(self):
        result = self.request('tool-and-text:true')
        self.assertEqual(result.error, 'ValueError')
        self.assertEqual(result.output_events, 0)
        self.assertEqual(summarize([result], 1)['succeeded'], 0)

    def test_partial_function_deltas_count_only_semantic_output(self):
        for calls in ([{'index':0,'id':'call_fixture','type':'function',
                        'function':{'name':'lookup','arguments':''}}],
                      [{'index':0,'function':{'name':None,'arguments':'{"x":'}}],
                      [{'index':0,'function':{'arguments':'1}'}}]):
            with self.subTest(calls=calls):
                result = self.request('text-field:tool_calls:' + json.dumps(calls))
                self.assertIsNone(result.error)
                self.assertEqual(result.output_events, 2)
                self.assertEqual(result.output_tokens, 7)
        for calls in (None, [], [{'index':0}],
                      [{'index':0,'id':'call_fixture','type':'function'}],
                      [{'index':0,'id':None,'type':None,'function':None}],
                      [{'index':0,'function':{'name':None,'arguments':''}}]):
            with self.subTest(calls=calls):
                result = self.request('text-field:tool_calls:' + json.dumps(calls))
                self.assertIsNone(result.error)
                self.assertEqual(result.output_events, 1)
                self.assertEqual(result.output_event_gaps_seconds, [])

    def test_terminal_chunk_can_include_last_output_before_usage(self):
        for reason in ('stop', 'length', 'tool_calls'):
            with self.subTest(reason=reason):
                result = self.request('finish-' + reason)
                self.assertIsNone(result.error)
                self.assertEqual(result.output_events, 3)
                self.assertEqual(result.output_tokens, 7)

    def test_http_error(self):
        self.assertEqual(self.request("http-error").error, "HTTP 503")

    def test_sse_multiline_crlf_and_comments(self):
        self.assertEqual(list(events([
            b": ping\r\n", b"event: message\r\n", b"data: first\r\n",
            b"data: second\r\n", b"\r\n", b"data: final\n",
        ])), ["first\nsecond", "final"])

    def test_sse_strict_termination_preserves_complete_events_and_tolerant_mode(self):
        wire = b'data: complete\r\n\r\ndata: partial\n'
        strict = events(io.BytesIO(wire), require_terminated=True)
        self.assertEqual(next(strict), 'complete')
        with self.assertRaisesRegex(ValueError, 'unterminated'):
            next(strict)
        self.assertEqual(list(events(io.BytesIO(wire))), ['complete', 'partial'])
        self.assertEqual(list(events(io.BytesIO(b'data: done\n\n'),
                                    require_terminated=True)), ['done'])
        self.assertEqual(list(events(io.BytesIO(b': keepalive\n'),
                                    require_terminated=True)), [])

    def test_sse_bounded_reads_and_line_rejection(self):
        class Reader(io.BytesIO):
            def readline(self, limit=-1):
                self.assert_limit = limit
                return super().readline(limit)
        reader = Reader(b'data: ' + b'x' * 1000)
        with self.assertRaisesRegex(ValueError, 'line byte limit'):
            list(events(reader, max_line_bytes=16))
        self.assertEqual(reader.assert_limit, 17)
        self.assertEqual(reader.tell(), 17)
        self.assertEqual(list(events(io.BytesIO(b'data: a\n\n'), max_line_bytes=8)), ['a'])
        result = self.request('sse-large')
        self.assertEqual(result.error, 'ValueError')
        self.assertEqual(result.output_events, 0)

    def test_sse_event_bounds_include_ignored_fields_and_reset(self):
        line = b'data: a\n'
        self.assertEqual(list(events([line, line, b'\n', line], max_event_bytes=16,
                                     max_event_lines=2)), ['a\na', 'a'])
        for kwargs in ({'max_event_bytes': 15}, {'max_event_lines': 1}):
            with self.assertRaisesRegex(ValueError, 'event limit'):
                list(events([line, line], **kwargs))
        with self.assertRaisesRegex(ValueError, 'event limit'):
            list(events([b': ignored\n'] * 3, max_event_lines=2))
        with self.assertRaisesRegex(ValueError, 'event limit'):
            list(events([b'event: ignored\n'], max_event_bytes=8))
        for value in (0, -1, True):
            with self.assertRaisesRegex(ValueError, 'invalid SSE'):
                list(events([], max_line_bytes=value))

    def test_throughput_uses_wall_time_and_successful_tokens(self):
        report = summarize([
            Result(0, "a", 2, 0.1, 20),
            Result(1, "a", 2, 0.2, 20),
            Result(2, "a", 1, 0.1, 5, "error"),
        ], 2)
        self.assertEqual(report["successful_output_tokens_per_second"], 20)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["first_output_seconds"]["p95"], 0.2)

    def test_cli_sequential_switches_and_json(self):
        process = subprocess.run([
            sys.executable, "-S", "-m", "dllm", "bench", "--base-url", self.url,
            "--models", "a", "b", "a", "--requests", "3", "--concurrency", "1",
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual([r["model"] for r in report["results"]], ["a", "b", "a"])
        self.assertEqual(report["summary"]["succeeded"], 3)
        self.assertEqual(report["summary"]["successful_output_tokens"], 21)

    def test_fixed_arrivals_drop_at_client_capacity_without_backlog(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def send(index, model):
            calls.append(index)
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test did not release request')
            return Result(index, model, .2, .1, 7)
        with ThreadPoolExecutor(1) as pool:
            run = pool.submit(fixed_arrivals, send, ['a', 'b'], 5, 1, 50)
            try:
                self.assertTrue(entered.wait(1))
                # All later arrivals are offered during the blocked request.
                time.sleep(.2)
                self.assertEqual(calls, [0])
            finally:
                release.set()
            results = sorted(run.result(1), key=lambda result: result.index)
        self.assertEqual(len(results), 5)
        self.assertEqual([r.model for r in results], ['a', 'b', 'a', 'b', 'a'])
        self.assertEqual([r.scheduled_offset_seconds for r in results], [0, .02, .04, .06, .08])
        self.assertEqual([r.error for r in results], [None]+['client capacity']*4)
        self.assertIsNotNone(results[0].dispatch_delay_seconds)
        self.assertTrue(all(r.dispatch_delay_seconds is None for r in results[1:]))
        self.assertEqual(summarize(results, .2)['client_capacity_drops'], 4)

    def test_gap_percentiles_preserve_unknown_and_exclude_failed_streams(self):
        good = Result(0, 'a', 1, .1, 7, output_events=4,
                      output_event_gaps_seconds=[.1, .2, .3])
        failed = Result(1, 'a', 1, .1, 7, 'HTTP 503',
                        output_event_gaps_seconds=[99])
        report = summarize([good, failed], 2)
        self.assertEqual(report['output_event_gaps_seconds'], {'p50': .2, 'p95': .3, 'p99': .3})
        self.assertEqual(report['max_output_event_gap_seconds_per_request']['p99'], .3)
        self.assertEqual(report['requests_with_output_gaps'], 1)
        self.assertEqual(report['http_failures'], 1)
        self.assertEqual(distribution(list(range(100)))['p99'], 98)
        self.assertIsNone(summarize([Result(0, 'a', 1, .1, 7)], 1)['output_event_gaps_seconds'])

    def test_output_sample_memory_is_bounded(self):
        with patch('dllm.bench.MAX_OUTPUT_EVENTS', 1):
            result = self.request('normal')
        self.assertEqual(result.error, 'ValueError')
        self.assertEqual(result.output_events, 1)
        self.assertFalse(result.output_event_gaps_seconds)

    def test_fixed_arrival_cli_reports_mode_and_usage(self):
        process = subprocess.run([
            sys.executable, '-S', '-m', 'dllm', 'bench', '--base-url', self.url,
            '--models', 'a', '--requests', '3', '--concurrency', '2', '--arrival-rate', '10',
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual(report['schema_version'], 3)
        self.assertEqual(report['workload']['arrival_mode'], 'fixed_rate')
        self.assertEqual(report['workload']['offered_requests_per_second'], 10)
        self.assertEqual(report['summary']['successful_output_tokens'], 21)
        self.assertEqual(report['summary']['requests_with_output_gaps'], 3)
        self.assertTrue(all(r['dispatch_delay_seconds'] is not None for r in report['results']))
        for invalid in ('0', '-1', 'nan', 'inf'):
            process = subprocess.run([sys.executable, '-S', '-m', 'dllm', 'bench',
                '--models', 'a', '--arrival-rate', invalid], capture_output=True, timeout=5)
            self.assertEqual(process.returncode, 2)

    def test_goodput_excludes_latency_tails_missing_measurements_and_errors(self):
        profile = dict(max_first_output_seconds=.2, max_output_event_gap_seconds=.1,
                       min_request_output_tokens_per_second=10)
        rows = [
            Result(0, 'a', 2, .2, 20, output_event_gaps_seconds=[.1]),
            Result(1, 'a', 2, .3, 20, output_event_gaps_seconds=[.1]),
            Result(2, 'a', 2, .1, 20, output_event_gaps_seconds=[.2]),
            Result(3, 'a', 2, .1, 10, output_event_gaps_seconds=[.1]),
            Result(4, 'a', 2, .1, None, output_event_gaps_seconds=[.1]),
            Result(5, 'a', 2, .1, 20),
            Result(6, 'a', 0, error='client capacity'),
            Result(7, 'a', 1, error='HTTP 503'),
        ]
        report = profile_summary(rows, 4, profile)
        self.assertEqual(report['counts'], {'passed': 1, 'failed': 3, 'unmeasurable': 2, 'error': 2})
        self.assertFalse(report['all_requests_passed'])
        self.assertEqual(report['qualifying_requests_per_second'], .25)
        self.assertEqual(report['qualifying_output_tokens_per_second'], 5)
        self.assertEqual(report['qualifying_output_tokens'], 20)
        self.assertEqual(evaluate_profile(rows[0], profile)['status'], 'passed')
        self.assertFalse(profile_summary([], 0, profile)['all_requests_passed'])
        self.assertIsNone(profile_summary(rows, 0, profile)['qualifying_requests_per_second'])

    def test_fixed_arrival_profile_includes_dispatch_delay(self):
        profile = dict(max_first_output_seconds=.5, max_output_event_gap_seconds=.1,
                       min_request_output_tokens_per_second=8)
        row = Result(0, 'a', 2, .25, 20, output_event_gaps_seconds=[.1])
        self.assertEqual(evaluate_profile(row, profile)['status'], 'passed')
        row.scheduled_offset_seconds = 100
        row.dispatch_delay_seconds = .25
        boundary = evaluate_profile(row, profile)
        self.assertEqual(boundary['status'], 'passed')
        self.assertEqual(boundary['first_output_seconds'], .5)
        self.assertEqual(boundary['request_seconds'], 2.25)
        self.assertEqual(boundary['timing_origin'], 'scheduled_arrival')
        row.dispatch_delay_seconds = 1
        result = evaluate_profile(row, profile)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reasons'], ['first output budget exceeded',
                                           'request-average token rate below budget'])
        self.assertEqual(result['maximum_output_event_gap_seconds'], .1)
        self.assertEqual(row.seconds, 2)  # Raw post-dispatch timing is preserved.
        self.assertEqual(row.first_output_seconds, .25)
        report = profile_summary([row, Result(1, 'a', 0, error='client capacity',
                                             scheduled_offset_seconds=101)], 4, profile)
        self.assertEqual(report['qualifying_output_tokens'], 0)
        self.assertEqual(report['counts'], dict(passed=0, failed=1, unmeasurable=0, error=1))

    def test_fixed_arrival_missing_or_invalid_dispatch_is_unmeasurable(self):
        profile = dict(max_first_output_seconds=1, max_output_event_gap_seconds=1,
                       min_request_output_tokens_per_second=1)
        row = Result(0, 'a', 2, .25, 20, output_event_gaps_seconds=[.1], scheduled_offset_seconds=0)
        for delay in (None, -1, float('nan'), float('inf'), True):
            row.dispatch_delay_seconds = delay
            self.assertEqual(evaluate_profile(row, profile)['status'], 'unmeasurable')
        row.dispatch_delay_seconds = 0
        self.assertEqual(evaluate_profile(row, profile)['status'], 'passed')
        row.scheduled_offset_seconds = -1
        self.assertEqual(evaluate_profile(row, profile)['status'], 'unmeasurable')

    def test_cli_profile_validation_and_failure_reporting(self):
        base = [sys.executable, '-S', '-m', 'dllm', 'bench', '--base-url', self.url,
                '--models', 'normal', '--requests', '1']
        for options in (['--slo-first-output', '1'], ['--slo-output-gap', 'nan'],
                        ['--slo-request-rate', '0']):
            result = subprocess.run(base+options, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
        options = ['--slo-first-output', '5', '--slo-output-gap', '5', '--slo-request-rate', '.001']
        result = subprocess.run(base+options, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['client_profile']['all_requests_passed'])
        self.assertEqual(report['workload']['profile_timing_origin'], 'request_start')
        fixed = subprocess.run(base+options+['--arrival-rate', '10'], capture_output=True, text=True, timeout=5)
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        report = json.loads(fixed.stdout)
        self.assertEqual(report['workload']['profile_timing_origin'], 'scheduled_arrival')
        raw = report['results'][0]
        evaluated = report['client_profile']['results'][0]
        self.assertAlmostEqual(evaluated['first_output_seconds'],
                               raw['first_output_seconds'] + raw['dispatch_delay_seconds'])
        result = subprocess.run(base+options+['--models', 'no-usage'], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report['summary']['succeeded'], 1)
        self.assertEqual(report['client_profile']['counts']['unmeasurable'], 1)
        self.assertEqual(report['client_profile']['qualifying_output_tokens'], 0)

    def test_cli_failure_exit_and_report(self):
        process = subprocess.run([
            sys.executable, "-S", "-m", "dllm", "bench", "--base-url", self.url,
            "--models", "http-error", "--requests", "1",
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
