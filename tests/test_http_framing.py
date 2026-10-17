"""Request framing must be complete before a runtime operation starts."""
import socket
import io
from types import SimpleNamespace
import threading
import time
import unittest
from http.client import HTTPResponse
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch

from damendaemon.http_auth import AuthPolicy, ClientKey
from damendaemon.server import handler_for


class HttpFramingTests(unittest.TestCase):
    def test_body_reader_restores_socket_timeout_on_success_and_failure(self):
        handler = self.server.RequestHandlerClass
        for payload in (b'{}', b'{'):
            with self.subTest(payload=payload):
                connection = Mock()
                connection.gettimeout.return_value = 7
                request = SimpleNamespace(connection=connection, rfile=io.BytesIO(payload))
                if len(payload) == 2:
                    self.assertEqual(handler.read_body(request, 2), b'{}')
                else:
                    with self.assertRaisesRegex(ValueError, 'incomplete'):
                        handler.read_body(request, 2)
                self.assertEqual(connection.settimeout.call_args.args, (7,))

    def test_continuing_body_progress_cannot_extend_total_deadline(self):
        from tests.test_http_auth import API_KEY, ADMIN_KEY
        client = ClientKey('slow-body', API_KEY, max_inflight=1)
        self.server.RequestHandlerClass = handler_for(self.runtime, AuthPolicy(
            admin_key=ADMIN_KEY, clients=(client,)))
        admission = self.server.RequestHandlerClass.client_admission
        reserved = threading.Event()
        reserve = admission._reserve
        def observe_reserve(*args, **kwargs):
            claim = reserve(*args, **kwargs)
            reserved.set()
            return claim
        admission._reserve = observe_reserve
        stop = threading.Event()
        with patch('damendaemon.server.BODY_READ_SECONDS', .15, create=True), \
                socket.create_connection(self.server.server_address, timeout=.8) as connection:
            connection.sendall(b'POST /v1/embeddings HTTP/1.1\r\n'
                b'Host: localhost\r\nContent-Type: application/json\r\n'
                + f'Authorization: Bearer {API_KEY}\r\n'.encode()
                + b'Content-Length: 100\r\n\r\n{}')
            def trickle():
                try:
                    for _ in range(98):
                        if stop.wait(.02):
                            return
                        connection.sendall(b' ')
                except OSError:
                    pass
            sender = threading.Thread(target=trickle)
            sender.start()
            try:
                self.assertTrue(reserved.wait(.8))
                started = time.monotonic()
                try:
                    received = connection.recv(1)
                except ConnectionResetError:
                    received = b''
                self.assertEqual(received, b'')
                self.assertLess(time.monotonic() - started, .8)
                self.runtime.embed.assert_not_called()
                self.assertEqual(admission.active, {})
                with admission.lease(['Bearer ' + API_KEY]):
                    self.assertEqual(admission.active, {'slow-body': 1})
            finally:
                stop.set()
                sender.join(1)

    def setUp(self):
        self.runtime = Mock()
        self.runtime.drain.return_value = {'draining': True}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0),
            handler_for(self.runtime, AuthPolicy()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, headers, body=b'{}', path='/v1/scheduler/drain'):
        with socket.create_connection(self.server.server_address, timeout=2) as connection:
            connection.sendall(f'POST {path} HTTP/1.1\r\n'.encode() +
                b'Host: localhost\r\nContent-Type: application/json\r\n' +
                headers + b'\r\n' + body)
            connection.shutdown(socket.SHUT_WR)
            response = HTTPResponse(connection)
            response.begin()
            payload = response.read()
            return response.status, payload

    def test_short_json_body_never_dispatches(self):
        self.assertEqual(self.request(b'Content-Length: 3\r\n')[0], 400)
        self.runtime.drain.assert_not_called()

    def test_ambiguous_and_unsupported_framing_never_dispatches(self):
        for headers in (
            b'Content-Length: 2\r\nContent-Length: 3\r\n',
            b'Content-Length: 2\r\nContent-Length: 2\r\n',
            b'Content-Length: 2\r\nTransfer-Encoding: chunked\r\n',
            b'Content-Length: 2\r\nTransfer-Encoding: identity\r\n',
            b'Content-Length: +2\r\n',
            b'Content-Length: 0_2\r\n',
        ):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(headers)[0], 400)
                self.runtime.drain.assert_not_called()

    def test_valid_body_dispatches_once(self):
        self.assertEqual(self.request(b'Content-Length: 2\r\n')[0], 200)
        self.runtime.drain.assert_called_once_with()

    def test_invalid_json_never_dispatches_management(self):
        self.runtime.manage.return_value = {}
        bodies = [
            b'{"model":"first","model":"second"}',
            b'{"model":"a","chat_template":{"x":1,"x":2}}',
            b'{"model":"a","chat_template":{"x":1,"\\u0078":2}}',
        ]
        for number in (b'NaN', b'Infinity', b'-Infinity', b'1e999', b'-1e999'):
            bodies.append(b'{"model":"a","chat_template":{"value":' + number + b'}}')
        bodies.append(b'{"model":"a","chat_template":' + b'[' * 10000 + b'0' + b']' * 10000 + b'}')
        for body in bodies:
            with self.subTest(body=body[:100]):
                status, _ = self.request(f'Content-Length: {len(body)}\r\n'.encode(),
                    body, '/v1/models/register')
                self.assertEqual(status, 400)
                self.runtime.manage.assert_not_called()

    def test_valid_nested_json_dispatches_and_model_recursion_remains_500(self):
        body = b'{"model":"a","chat_template":{"value":1.25,"text":"NaN"}}'
        headers = f'Content-Length: {len(body)}\r\n'.encode()
        self.runtime.manage.return_value = {}
        self.assertEqual(self.request(headers, body, '/v1/models/register')[0], 200)
        self.runtime.manage.assert_called_once_with('register', model='a',
            chat_template={'value': 1.25, 'text': 'NaN'})
        self.runtime.manage.side_effect = RecursionError('runtime failure')
        with patch('damendaemon.server.traceback.print_exc'):
            self.assertEqual(self.request(headers, body, '/v1/models/register')[0], 500)
