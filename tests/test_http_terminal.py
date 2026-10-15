"""Completion publication must follow terminal validation and runtime cleanup."""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.bench import events
from dllm.http_auth import AuthPolicy
from dllm.server import handler_for


class HttpTerminalTests(unittest.TestCase):
    def setUp(self):
        self.case = 'valid'
        self.cleaned = threading.Event()
        self.final = dict(finish_reason='stop', usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2), timing={})
        def generate(*args, **kwargs):
            try:
                if self.case == 'empty': return
                yield dict(text='partial', token=1)
                if self.case == 'missing': return
                if self.case == 'invalid-final': yield dict(self.final, finish_reason=None)
                elif self.case == 'missing-metadata': yield dict(finish_reason='stop')
                else: yield self.final
                if self.case == 'duplicate': yield self.final
                if self.case == 'late-output': yield dict(text='late', token=2)
            finally:
                self.cleaned.set()
                if self.case == 'cleanup-fails': raise RuntimeError('injected cleanup failure')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0),
            handler_for(SimpleNamespace(generate=generate), AuthPolicy()))
        admission = self.server.RequestHandlerClass.client_admission
        original_lease = admission.output_lease
        @contextmanager
        def output_lease(*args, **kwargs):
            with original_lease(*args, **kwargs):
                yield
            if self.case == 'output-lease-fails':
                raise RuntimeError('injected output lease cleanup failure')
        admission.output_lease = output_lease
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, stream):
        payload = dict(model='a', messages=[dict(role='user', content='hi')], stream=stream,
                       stream_options={'include_usage': True})
        return urlopen(Request(f'http://127.0.0.1:{self.server.server_port}/v1/chat/completions',
            json.dumps(payload).encode(), {'Content-Type': 'application/json'}), timeout=3)

    def test_missing_or_invalid_terminal_never_publishes_stream_success(self):
        with patch('dllm.server.traceback.print_exc'):
            for case in ('missing','duplicate','late-output','cleanup-fails',
                         'invalid-final','missing-metadata','output-lease-fails'):
                with self.subTest(case=case):
                    self.case=case
                    self.cleaned.clear()
                    with self.request(True) as response:
                        chunks=list(events(response, require_terminated=True))
                    self.assertTrue(self.cleaned.is_set())
                    self.assertNotIn('[DONE]',chunks)
                    payloads=[json.loads(chunk) for chunk in chunks]
                    self.assertTrue(any('error' in item for item in payloads))
                    self.assertFalse(any('usage' in item for item in payloads))
                    self.assertFalse(any(choice.get('finish_reason') for item in payloads
                                         for choice in item.get('choices',[])))

    def test_missing_final_is_server_error_including_empty_stream(self):
        with patch('dllm.server.traceback.print_exc'):
            for case, stream in (('missing',False),('empty',False),('empty',True)):
                with self.subTest(case=case,stream=stream):
                    self.case=case
                    with self.assertRaises(HTTPError) as raised:
                        self.request(stream)
                    self.assertEqual(raised.exception.code,500)
                    raised.exception.close()
                    self.assertTrue(self.cleaned.is_set())

    def test_valid_stream_retains_one_finish_usage_and_done(self):
        with self.request(True) as response:
            chunks=list(events(response, require_terminated=True))
        self.assertEqual(chunks[-1],'[DONE]')
        payloads=[json.loads(chunk) for chunk in chunks[:-1]]
        self.assertEqual(sum('usage' in item for item in payloads),1)
        finishes=[choice['finish_reason'] for item in payloads for choice in item.get('choices',[])
                  if choice.get('finish_reason')]
        self.assertEqual(finishes,['stop'])
        self.assertTrue(self.cleaned.is_set())
