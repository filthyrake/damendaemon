"""Connection bounds must fail closed before constructing server resources."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler

from dllm.server import BoundedHTTPServer


class HttpCapacityTests(unittest.TestCase):
    def test_invalid_capacity_never_binds_and_closes_reservation_owner(self):
        for capacity in (0, -1, True, False, .5, 1.5, float('nan'),
                         float('inf'), None, '2'):
            with self.subTest(capacity=capacity):
                admission = Mock(reservation_store=object())
                handler = SimpleNamespace(client_admission=admission)
                with patch('dllm.server.ThreadingHTTPServer.__init__') as initialize:
                    with self.assertRaises(ValueError):
                        BoundedHTTPServer(('127.0.0.1', 0), handler, capacity)
                    initialize.assert_not_called()
                admission.close.assert_called_once_with()

    def test_valid_capacity_is_exhaustible_and_reusable(self):
        server = BoundedHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler, 2)
        self.addCleanup(server.server_close)
        self.assertTrue(server.connections.acquire(blocking=False))
        self.assertTrue(server.connections.acquire(blocking=False))
        self.assertFalse(server.connections.acquire(blocking=False))
        server.connections.release()
        self.assertTrue(server.connections.acquire(blocking=False))
        self.assertFalse(server.connections.acquire(blocking=False))
        server.connections.release()
        server.connections.release()
