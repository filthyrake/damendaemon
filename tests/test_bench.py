from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
import unittest
import time
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from dllm.bench import Result, events, request_one, summarize, llama_context, fixed_arrivals, distribution


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
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "several tokens in one chunk"}}]},
            {"choices": [{"delta": {"reasoning_content": "another output event"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        if model != "no-usage":
            chunks.append({"choices": [], "usage": {"completion_tokens": 7}})
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if model != "truncated":
            self.wfile.write(b"data: [DONE]\n\n")


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

    def test_unknown_tokens_are_not_invented(self):
        result = self.request("no-usage")
        self.assertIsNone(result.error)
        self.assertIsNone(summarize([result], 1)["successful_output_tokens_per_second"])

    def test_truncated_stream_fails_even_with_usage(self):
        result = self.request("truncated")
        self.assertIsNotNone(result.error)
        self.assertEqual(summarize([result], 1)["succeeded"], 0)

    def test_http_error(self):
        self.assertEqual(self.request("http-error").error, "HTTP 503")

    def test_sse_multiline_crlf_and_comments(self):
        self.assertEqual(list(events([
            b": ping\r\n", b"event: message\r\n", b"data: first\r\n",
            b"data: second\r\n", b"\r\n", b"data: final\n",
        ])), ["first\nsecond", "final"])

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
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(report['workload']['arrival_mode'], 'fixed_rate')
        self.assertEqual(report['workload']['offered_requests_per_second'], 10)
        self.assertEqual(report['summary']['successful_output_tokens'], 21)
        self.assertEqual(report['summary']['requests_with_output_gaps'], 3)
        self.assertTrue(all(r['dispatch_delay_seconds'] is not None for r in report['results']))
        for invalid in ('0', '-1', 'nan', 'inf'):
            process = subprocess.run([sys.executable, '-S', '-m', 'dllm', 'bench',
                '--models', 'a', '--arrival-rate', invalid], capture_output=True, timeout=5)
            self.assertEqual(process.returncode, 2)

    def test_cli_failure_exit_and_report(self):
        process = subprocess.run([
            sys.executable, "-S", "-m", "dllm", "bench", "--base-url", self.url,
            "--models", "http-error", "--requests", "1",
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
