"""Quota timeout validation uses no credentials, TLS context or network."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from damendaemon.remote_quota_transport import _timeout
from damendaemon.remote_quota_config import frontend_client, authority_listener
from scripts.serve_quota_authority import main


class QuotaTimeoutTests(unittest.TestCase):
    def test_bounds_before_float_conversion(self):
        for value in (10**400, -(10**400), True, False, None, '1', 0, -1, 31,
                      float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'quota timeout'):
                _timeout(value)
        for value in (1,30,.001,29.5):
            self.assertEqual(_timeout(value),float(value))
            self.assertIs(type(_timeout(value)),float)

    def test_private_configs_reject_before_tls_or_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            tls=dict(ca_file='/unused/ca',cert_file='/unused/cert',key_file='/unused/key')
            front=dict(tls,address='127.0.0.1',port=1,server_hostname='localhost',
                       ledger='0'*32,owner='0'*32,server_pin='0'*64)
            listener=dict(tls,address='127.0.0.1',port=0,client_pins=['0'*64])
            for read,config in ((frontend_client,front),(authority_listener,listener)):
                for value in (10**400,-(10**400)):
                    path.write_text(json.dumps(dict(config,timeout=value)));path.chmod(0o600)
                    with self.subTest(reader=read.__name__,timeout=value), \
                            patch('damendaemon.remote_quota_config._tls') as context, \
                            patch('socket.create_connection') as connect:
                        with self.assertRaisesRegex(ValueError,'quota timeout'):
                            read(path)
                        context.assert_not_called();connect.assert_not_called()
                    if read is frontend_client:
                        error=io.StringIO()
                        with redirect_stderr(error), self.assertRaises(SystemExit) as exited:
                            main(['status','--config',str(path)])
                        self.assertEqual(exited.exception.code,2)
                        self.assertIn('quota command failed',error.getvalue())
                        self.assertNotIn('Traceback',error.getvalue())
                        self.assertNotIn(str(value),error.getvalue())
