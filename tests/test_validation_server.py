"""Verifier HTTP ownership tests; no model execution or remote services."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from scripts.validation_server import owned_http_server


class ValidationServerTests(unittest.TestCase):
    def owners(self):
        runtime=SimpleNamespace(close=Mock())
        server=SimpleNamespace(serve_forever=Mock(),shutdown=Mock(),server_close=Mock())
        thread=Mock();thread.is_alive.return_value=False
        return runtime,server,thread

    def test_bind_failure_still_closes_runtime(self):
        runtime,server,thread=self.owners()
        with patch('scripts.validation_server.threading.Thread') as construct:
            with self.assertRaisesRegex(OSError,'bind'):
                with owned_http_server(runtime,server_factory=Mock(side_effect=OSError('bind'))):
                    self.fail('startup unexpectedly succeeded')
            construct.assert_not_called()
        runtime.close.assert_called_once_with()

    def test_thread_start_failure_does_not_wait_for_unstarted_server(self):
        runtime,server,thread=self.owners();thread.start.side_effect=RuntimeError('start')
        with patch('scripts.validation_server.threading.Thread',return_value=thread):
            with self.assertRaisesRegex(RuntimeError,'start'):
                with owned_http_server(runtime,server_factory=Mock(return_value=server)):pass
        runtime.close.assert_called_once_with();server.server_close.assert_called_once_with()
        server.shutdown.assert_not_called();thread.join.assert_not_called()

    def test_cleanup_attempts_every_owner_and_preserves_primary(self):
        for primary in (False,True):
            for failing in ('shutdown','runtime','server_close'):
                with self.subTest(primary=primary,failing=failing):
                    runtime,server,thread=self.owners();order=[]
                    def action(name):
                        def run():
                            order.append(name)
                            if name==failing:raise OSError('cleanup fixture')
                        return run
                    server.shutdown.side_effect=action('shutdown')
                    runtime.close.side_effect=action('runtime')
                    server.server_close.side_effect=action('server_close')
                    original=ValueError('body failed')
                    with patch('scripts.validation_server.threading.Thread',return_value=thread):
                        with self.assertRaises(ValueError if primary else ExceptionGroup) as caught:
                            with owned_http_server(runtime,server_factory=Mock(return_value=server)):
                                if primary:raise original
                    self.assertEqual(order,['shutdown','runtime','server_close'])
                    thread.join.assert_called_once_with(timeout=5)
                    if primary:
                        self.assertIs(caught.exception,original)
                        self.assertIn('Verifier cleanup also failed',original.__notes__[0])

    def test_actual_loopback_server_stops_and_closes_runtime(self):
        import json
        from urllib.request import urlopen
        runtime=SimpleNamespace(close=Mock())
        with owned_http_server(runtime) as server:
            with urlopen(f'http://127.0.0.1:{server.server_port}/health',timeout=2) as response:
                self.assertEqual(json.load(response),{'status':'ok'})
            runtime.close.assert_not_called()
        runtime.close.assert_called_once_with()
        self.assertEqual(server.socket.fileno(),-1)

    def test_thread_that_does_not_stop_is_a_cleanup_failure(self):
        runtime,server,thread=self.owners();thread.is_alive.return_value=True
        with patch('scripts.validation_server.threading.Thread',return_value=thread):
            with self.assertRaisesRegex(ExceptionGroup,'verifier cleanup failed') as caught:
                with owned_http_server(runtime,server_factory=Mock(return_value=server)):pass
        self.assertIn('thread did not stop',str(caught.exception.exceptions[0]))
        runtime.close.assert_called_once_with();server.server_close.assert_called_once_with()

    def test_normal_context_keeps_ownership_until_exit(self):
        runtime,server,thread=self.owners()
        with patch('scripts.validation_server.threading.Thread',return_value=thread):
            with owned_http_server(runtime,server_factory=Mock(return_value=server)) as actual:
                self.assertIs(actual,server);runtime.close.assert_not_called()
            runtime.close.assert_called_once_with()
            server.shutdown.assert_called_once_with();server.server_close.assert_called_once_with()
