from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
import unittest

from dllm.bench import Result, events, request_one, summarize


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

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

    def test_stream_usage_is_not_chunk_count(self):
        result = self.request("normal")
        self.assertIsNone(result.error)
        self.assertEqual(result.output_tokens, 7)
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

    def test_cli_failure_exit_and_report(self):
        process = subprocess.run([
            sys.executable, "-S", "-m", "dllm", "bench", "--base-url", self.url,
            "--models", "http-error", "--requests", "1",
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
