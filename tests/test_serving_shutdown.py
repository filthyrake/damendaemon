from contextlib import redirect_stdout
import io
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import unittest
from unittest.mock import Mock,patch
from dllm.server import serve
from tests import test_quota_commands as quota,test_glm_host_http_command as native,test_cpu_host_topology_command as topology
from tests.test_distributed_drain import wait_for


class ServingShutdownTests(unittest.TestCase):
    def setUp(self):
        self.quota=quota.QuotaCommandTests();self.quota.setUp();self.addCleanup(self.quota.doCleanups)

    def quota_env(self):
        q=self.quota
        return q.env|{'DLLM_REMOTE_QUOTA_CONFIG':str(q.config),'DLLM_CLIENT_KEYS':json.dumps(
            {'alice':dict(key=q.key,max_inflight=1,max_reserved_output_tokens=8)})}

    def hold(self,url,route='/v1/chat/completions'):
        client=self.quota.client()
        connection=socket.create_connection(('127.0.0.1',int(url.rsplit(':',1)[1])),timeout=5)
        self.addCleanup(connection.close)
        connection.sendall(('POST '+route+' HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer '+self.quota.key+
            '\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{').encode())
        wait_for(lambda:client.call(client.request('status'))['claims']==1)
        return connection

    def end_request(self,connection,process,*,success=True):
        with self.assertRaises(subprocess.TimeoutExpired):process.wait(timeout=.5)
        connection.sendall(b'}')
        while connection.recv(65536):pass
        out,err=process.communicate(timeout=12)
        if success:self.assertEqual(process.returncode,0,err)
        else:self.assertNotEqual(process.returncode,0)
        self.assertNotIn(self.quota.key,out+err)
        self.assertNotIn(self.quota.env['DLLM_REMOTE_QUOTA_OWNER_KEY'],out+err)
        return out,err

    def test_ordinary_serving_owner_sigterm_waits_for_remote_request_release(self):
        authority=self.quota.bootstrap()
        source="""from dllm.server import serve
from tests.test_server import FakeRuntime
class Runtime(FakeRuntime):
    def close(self): print('runtime_closed',flush=True)
serve(Runtime(),0)
print('stopped',flush=True)
"""
        process=subprocess.Popen([sys.executable,'-c',source],env=self.quota_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.quota.processes.append(process)
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout,selectors.EVENT_READ);self.assertTrue(ready.select(12))
        line=process.stdout.readline();self.assertTrue(line.startswith('dllm listening on '),line)
        connection=self.hold(line.strip().split()[-1],'/v1/completions')
        process.send_signal(signal.SIGTERM);out,_=self.end_request(connection,process)
        self.assertEqual(out.splitlines(),['runtime_closed','stopped'])
        client=self.quota.client();self.assertEqual(client.call(client.request('status'))['claims'],0)
        self.quota.stop(authority)

    def native_service(self):
        fixture=native.GlmHostHttpCommandTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.environment.update(self.quota_env())
        workers=fixture.workers('fp8_reference');coordinator=fixture.launch(fixture.args('serve'))
        ready=fixture.event(coordinator);self.assertEqual(ready['event'],'http_ready')
        return fixture,workers,coordinator,ready['url']

    def test_native_host_stop_waits_for_http_lease_and_acknowledged_peer_cleanup(self):
        authority=self.quota.bootstrap();fixture,workers,coordinator,url=self.native_service()
        connection=self.hold(url);coordinator.send_signal(signal.SIGTERM)
        out,_=self.end_request(connection,coordinator)
        self.assertTrue(json.loads(out)['graceful_stop']);fixture.finish_workers(workers)
        client=self.quota.client();self.assertEqual(client.call(client.request('status'))['claims'],0)
        self.quota.stop(authority)

    def test_native_host_failed_quota_cleanup_cannot_report_graceful_stop(self):
        authority=self.quota.bootstrap();fixture,workers,coordinator,url=self.native_service()
        connection=self.hold(url);self.quota.stop(authority);coordinator.send_signal(signal.SIGTERM)
        out,_=self.end_request(connection,coordinator,success=False)
        self.assertNotIn('graceful_stop',out);fixture.finish_workers(workers)
        authority=self.quota.start();client=self.quota.client()
        self.assertEqual(client.call(client.request('status'))['claims'],1);self.quota.stop(authority)

    def test_cpu_topology_sigterm_reports_stop_only_after_http_cleanup(self):
        authority=self.quota.bootstrap()
        fixture=topology.CpuHostTopologyCommandTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.fixture.environment.update(self.quota_env())
        workers=fixture.workers();coordinator=fixture.launch(fixture.args('serve'))
        ready=fixture.event(coordinator);connection=self.hold(ready['url'],'/v1/completions')
        coordinator.send_signal(signal.SIGTERM);out,_=self.end_request(connection,coordinator)
        self.assertEqual(json.loads(out),dict(event='stopped',scope='cpu_host_reference'))
        for worker in workers:
            out,err=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,err)
            self.assertNotIn('"event": "stopped"',out)
        client=self.quota.client();self.assertEqual(client.call(client.request('status'))['claims'],0)
        self.quota.stop(authority)

    def test_ordinary_bind_failure_and_cleanup_errors_preserve_all_owner_attempts(self):
        runtime=Mock()
        with patch('dllm.server.handler_for',side_effect=ValueError('invalid configuration')):
            with self.assertRaisesRegex(ValueError,'invalid configuration'):serve(runtime,0)
        runtime.close.assert_called_once()
        server=Mock(server_port=1234);server.handle_request.side_effect=KeyboardInterrupt
        runtime=Mock();runtime.close.side_effect=RuntimeError('retained runtime')
        with patch('dllm.server.handler_for'),patch('dllm.server.BoundedHTTPServer',return_value=server),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'retained runtime'):serve(runtime,0)
        server.server_close.assert_called_once();self.assertFalse(server.daemon_threads);self.assertTrue(server.block_on_close)

    def test_distributed_command_control_cleanup_attempts_every_owner_after_failure(self):
        # Lifecycle control only: no StagePair construction or device operation.
        from dllm import distributed_server_cli as command
        pair=Mock(rank=0);runtime=Mock();runtime.run.side_effect=RuntimeError('run failed')
        runtime.close.side_effect=RuntimeError('runtime retained')
        server=Mock(server_port=1234);server.server_close.side_effect=RuntimeError('quota retained')
        pair.stop.side_effect=RuntimeError('stop failed')
        with (patch.dict(os.environ,{},clear=True),patch('dllm.distributed.StagePair',return_value=pair),
              patch('dllm.distributed_serving.DistributedRuntime',return_value=runtime),
              patch('dllm.server.handler_for',return_value=type('Handler',(),{})),
              patch('dllm.server.BoundedHTTPServer',return_value=server),redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError,'run failed') as error:command.main(['unused'])
        self.assertTrue(error.exception.__notes__)
        runtime.close.assert_called_once();server.server_close.assert_called_once()
        pair.stop.assert_called_once();pair.close.assert_called_once();self.assertFalse(server.daemon_threads)


if __name__=='__main__':unittest.main()
