"""Request framing must be complete before a runtime operation starts."""
import socket
import threading
import unittest
from http.client import HTTPResponse
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

from dllm.http_auth import AuthPolicy
from dllm.server import handler_for


class HttpFramingTests(unittest.TestCase):
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

    def request(self, headers, body=b'{}'):
        with socket.create_connection(self.server.server_address, timeout=2) as connection:
            connection.sendall(b'POST /v1/scheduler/drain HTTP/1.1\r\n'
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
