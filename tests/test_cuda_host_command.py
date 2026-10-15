from contextlib import redirect_stderr
import io
import json
import unittest
from unittest.mock import patch
from scripts import run_cuda_host_topology as cli,run_cpu_host_topology as shared
from tests import test_glm_native_host_command as native


class CudaHostCommandTests(unittest.TestCase):
    setUp=native.GlmNativeHostCommandTests.setUp
    snapshot=native.GlmNativeHostCommandTests.snapshot

    def args(self):
        self.peers.write_text(json.dumps([dict(stage=i,address='127.0.0.1',port=1234+i,
            server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]))
        return ['--family','glm',*native.GlmNativeHostCommandTests.args(self,'serve'),'--device','cuda:0']

    def test_invalid_device_mode_and_bounds_precede_metadata_and_network(self):
        for extra in (['--device','cpu'],['--device','cuda'],['--allocator-mib','128'],['--device-weight-mib','0'],
                      ['--transfer-mib','0'],['--cpu-threads','17']):
            with patch.object(shared,'prepare',side_effect=AssertionError('metadata')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:cli.main(self.args()+extra)
                self.assertEqual(caught.exception.code,2)
        args=self.args();args.remove('--native-chat')
        with patch.object(shared,'prepare',side_effect=AssertionError('metadata')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(args)

    def test_unavailable_device_precedes_connection(self):
        with patch('scripts.cuda_host_execution.configure',side_effect=ValueError('unavailable CUDA')),\
                patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:cli.main(self.args())
            self.assertEqual(caught.exception.code,2)

    def test_cpu_command_does_not_accept_cuda_options(self):
        from scripts.run_glm_host_topology import main
        args=self.args()[2:]
        with patch.object(shared,'run',side_effect=AssertionError('run')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):main(args)

    def test_flash_image_mode_rejects_before_metadata(self):
        args=self.args();args[1]='flash';args+=['--flash-images']
        with patch.object(shared,'prepare',side_effect=AssertionError('metadata')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:cli.main(args)
            self.assertEqual(caught.exception.code,2)
