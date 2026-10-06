"""Measurement integrity only; these fixtures are not GPU performance proof."""
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('modern_http', Path(__file__).parents[1] / 'scripts/benchmark_modern_http.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Response(io.BytesIO):
    headers = Message()
    headers['Content-Type'] = 'text/event-stream'


class BenchmarkIntegrity(unittest.TestCase):
    def run_response(self, output=128, prompt=2, done=True, empty=False):
        chunks = [dict(choices=[dict(text='' if empty else 'hello')]),
                  dict(choices=[], usage=dict(prompt_tokens=prompt, completion_tokens=output))]
        stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
        if done:
            stream += 'data: [DONE]\n\n'
        with patch.object(benchmark, 'urlopen', return_value=Response(stream.encode())) as post:
            result = benchmark.request('http://127.0.0.1:1', 'a', [1, 2], 'dllm', [3], threading.Barrier(1))
            body = json.loads(post.call_args.args[0].data)
            self.assertEqual(body['prompt'], [1, 2])
            self.assertEqual(body['logit_bias'], {'3': -100})
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


if __name__ == '__main__':
    unittest.main()
