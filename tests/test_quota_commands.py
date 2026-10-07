import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from dllm.client_admission import ClientAdmission
from dllm.http_auth import AuthPolicy,ClientKey
from dllm.host_tls import certificate_pin
from dllm.remote_quota_config import private_json,frontend_client,authority_policy,authority_listener,recovery_request
from dllm.shared_client_reservations import SharedReservationUnavailable
from tests import tls_credentials,test_serve_cpu_reference as reference
from tests.test_distributed_drain import wait_for


class QuotaCommandTests(unittest.TestCase):
    event=reference.ReferenceCommandTests.event

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.env={k:v for k,v in os.environ.items() if not k.startswith('DLLM_')}
        self.env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',DLLM_REMOTE_QUOTA_OWNER_KEY=secrets.token_hex(32))
        self.owner=secrets.token_hex(16);self.key=secrets.token_urlsafe(32)
        self.client_policy=ClientKey('alice',self.key,1,max_reserved_output_tokens=8)
        self.policy=self.write('policy.json',dict(path=str(self.path/'ledger.sqlite'),clients=[dict(name='alice',max_inflight=1,max_reserved_output_tokens=8)]))
        tls=tls_credentials.credentials()
        self.listener=dict(address='127.0.0.1',port=0,client_pins=[certificate_pin(tls/'client.pem').hex()],
            ca_file=str(tls/'ca.pem'),cert_file=str(tls/'server.pem'),key_file=str(tls/'server.key'),timeout=2)
        self.listen=self.write('listen.json',self.listener)
        self.front=dict(address='127.0.0.1',port=1,server_hostname='localhost',ledger=secrets.token_hex(16),owner=self.owner,
            server_pin=certificate_pin(tls/'server.pem').hex(),ca_file=str(tls/'ca.pem'),cert_file=str(tls/'client.pem'),key_file=str(tls/'client.key'),timeout=2)
        self.config=self.write('frontend.json',self.front)
        self.processes=[]
        self.addCleanup(self.cleanup_processes)

    def write(self,name,value):
        path=self.path/name;path.write_text(json.dumps(value));path.chmod(0o600);return path

    def cleanup_processes(self):
        for process in reversed(self.processes):
            if process.poll() is None:process.kill()
            process.communicate(timeout=10)

    def command(self,*args,success=True):
        result=subprocess.run([sys.executable,'-m','scripts.serve_quota_authority',*map(str,args)],
            env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
        if success:self.assertEqual(result.returncode,0,result.stderr)
        else:self.assertNotEqual(result.returncode,0)
        self.assertNotIn(self.env['DLLM_REMOTE_QUOTA_OWNER_KEY'],result.stdout+result.stderr)
        self.assertNotIn(self.key,result.stdout+result.stderr)
        return result

    def start(self):
        process=subprocess.Popen([sys.executable,'-m','scripts.serve_quota_authority','serve','--policy',str(self.policy),'--listen',str(self.listen)],
            env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.processes.append(process);ready=self.event(process)
        self.assertEqual(ready['event'],'quota_ready')
        self.front.update(port=ready['port'],ledger=ready['ledger']);self.write('frontend.json',self.front)
        return process

    def stop(self,process,scope='quota_authority'):
        process.send_signal(signal.SIGTERM);out,err=process.communicate(timeout=15)
        self.assertEqual(process.returncode,0,err)
        events=[json.loads(line) for line in out.splitlines()]
        self.assertIn(dict(event='stopped',scope=scope,**({'remote_claims_retained':True} if scope=='quota_authority' else {})),events)

    def bootstrap(self):
        self.command('init','--policy',self.policy)
        self.command('register','--policy',self.policy,'--owner',self.owner,'--client','alice')
        return self.start()

    def client(self):
        with patch.dict(os.environ,self.env,clear=True):return frontend_client(self.config)

    def operation(self,sequence,command,**extra):
        return self.write('operation.json',dict(ledger=self.front['ledger'],owner=self.owner,sequence=sequence,
            operation_id=secrets.token_hex(16),command=command,**extra))

    def test_private_bounded_configuration_rejects_duplicates_links_and_public_inputs(self):
        for value in ('[]','{"duplicate":1,"duplicate":2}','{"value":NaN}','x'*65537):
            self.config.write_text(value)
            with self.assertRaises(ValueError):private_json(self.config)
        self.write('frontend.json',self.front);self.config.chmod(0o644)
        with self.assertRaises(ValueError):private_json(self.config)
        self.config.chmod(0o600)
        alias=self.path/'alias';alias.symlink_to(self.config)
        with self.assertRaises(ValueError):private_json(alias)
        alias.unlink();os.link(self.config,alias)
        with self.assertRaises(ValueError):private_json(self.config)
        alias.unlink()
        with self.assertRaises(ValueError):private_json('relative.json')
        fifo=self.path/'fifo';os.mkfifo(fifo,0o600)
        with self.assertRaises(ValueError):private_json(fifo)

    def test_strict_policy_tls_and_endpoint_configuration(self):
        for change in (dict(address='localhost'),dict(port=True),dict(timeout=31),dict(server_pin='bad'),
                       dict(owner='bad'),dict(unknown=True),dict(ca_file='relative.pem'),dict(server_hostname=None)):
            self.write('frontend.json',self.front|change)
            with patch.dict(os.environ,self.env,clear=True),self.assertRaises(ValueError):frontend_client(self.config)
        for change in (dict(client_pins=[]),dict(client_pins=[self.listener['client_pins'][0]]*2),dict(max_connections=129)):
            self.write('listen.json',self.listener|change)
            with self.assertRaises(ValueError):authority_listener(self.listen)
        raw=private_json(self.policy);raw['clients'][0]['key']=self.key;self.write('policy.json',raw)
        with self.assertRaises(ValueError):authority_policy(self.policy)
        self.write('frontend.json',self.front)
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):frontend_client(self.config)

    def test_explicit_bootstrap_registration_restart_and_authenticated_recovery(self):
        self.command('serve','--policy',self.policy,'--listen',self.listen,success=False)
        self.assertFalse((self.path/'ledger.sqlite').exists())
        authority=self.bootstrap();client=self.client()
        request=client.request('execute',sequence=1,command=dict(op='reserve',client='alice',requests=1,outputs=8))
        claim=client.call(request)['claim']
        status=json.loads(self.command('status','--config',self.config).stdout);self.assertEqual(status['claims'],1)
        self.stop(authority)
        authority=self.start();client=self.client()
        self.assertEqual(client.call(request)['claim'],claim)
        rows=json.loads(self.command('claims','--config',self.config,'--limit','1').stdout)
        self.assertEqual(rows['claims'][0]['claim'],claim)
        with patch.dict(os.environ,self.env|{'DLLM_REMOTE_QUOTA_CONFIG':str(self.config)},clear=True):
            with self.assertRaises(SharedReservationUnavailable):ClientAdmission(AuthPolicy(clients=(self.client_policy,)))
        operation=self.operation(2,dict(op='release',claim=claim))
        first=self.command('recover','--config',self.config,'--operation',operation).stdout
        self.assertEqual(self.command('recover','--config',self.config,'--operation',operation).stdout,first)
        # A new nonce at an already-used sequence cannot stand in for that retry.
        self.operation(2,dict(op='release',claim=claim))
        self.command('recover','--config',self.config,'--operation',operation,success=False)
        self.operation(3,dict(op='retire'));self.command('recover','--config',self.config,'--operation',operation)
        self.command('register','--policy',self.policy,'--owner',self.owner,'--client','alice',success=False)
        self.stop(authority)
        self.assertNotIn(self.env['DLLM_REMOTE_QUOTA_OWNER_KEY'].encode(),(self.path/'ledger.sqlite').read_bytes())

    def test_recovery_refuses_reserve_wrong_identity_and_changed_shape(self):
        client=self.client()
        for command in (dict(op='reserve',client='alice',requests=1,outputs=0),dict(op='release',claim='bad'),dict(op='retire',extra=True)):
            path=self.operation(1,command)
            with self.assertRaises(ValueError):recovery_request(client,path)
        path=self.operation(1,dict(op='retire'));raw=private_json(path);raw['owner']=secrets.token_hex(16);self.write('operation.json',raw)
        with self.assertRaises(ValueError):recovery_request(client,path)

    def test_frontend_opt_in_defaults_conflict_and_matching_policy(self):
        with patch.dict(os.environ,{},clear=True):
            admission=ClientAdmission(AuthPolicy(clients=(self.client_policy,)));self.assertIsNone(admission.reservation_store);admission.close()
        with patch.dict(os.environ,dict(DLLM_REMOTE_QUOTA_CONFIG=str(self.config),DLLM_CLIENT_RESERVATION_DB=str(self.path/'unused.sqlite')),clear=True):
            with self.assertRaises(ValueError):ClientAdmission(AuthPolicy(clients=(self.client_policy,)))
        self.assertFalse((self.path/'unused.sqlite').exists())
        authority=self.bootstrap()
        with patch.dict(os.environ,self.env|{'DLLM_REMOTE_QUOTA_CONFIG':str(self.config)},clear=True):
            admission=ClientAdmission(AuthPolicy(clients=(self.client_policy,)))
            with admission.lease(['Bearer '+self.key]):self.assertEqual(self.client().call(self.client().request('status'))['claims'],1)
            admission.close()
            wrong=ClientKey('alice',self.key,2,max_reserved_output_tokens=8)
            with self.assertRaises(SharedReservationUnavailable):ClientAdmission(AuthPolicy(clients=(wrong,)))
        self.stop(authority)

    def test_cpu_command_http_and_sigterm_wait_for_inflight_remote_lease(self):
        authority=self.bootstrap();client=self.client()
        fixture=reference.ReferenceCommandTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        snapshot=fixture.snapshot('glm')
        env=self.env|{'DLLM_REMOTE_QUOTA_CONFIG':str(self.config),
            'DLLM_CLIENT_KEYS':json.dumps({'alice':dict(key=self.key,max_inflight=1,max_reserved_output_tokens=8)})}
        process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','glm',str(snapshot),'--context','256','--port','0'],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.processes.append(process);ready=self.event(process)
        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'}
        body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=1)
        for stream in (False,True):
            with urlopen(Request(ready['url']+'/v1/chat/completions',json.dumps(body|{'stream':stream}).encode(),headers),timeout=10) as response:
                data=response.read();self.assertEqual(response.status,200)
                if stream:self.assertIn(b'[DONE]',data)
                else:self.assertEqual(json.loads(data)['choices'][0]['finish_reason'],'length')
        wait_for(lambda:client.call(client.request('status'))['claims']==0)
        address=('127.0.0.1',int(ready['url'].rsplit(':',1)[1]))
        connection=socket.create_connection(address,timeout=5);self.addCleanup(connection.close)
        data=json.dumps(body).encode()
        connection.sendall(('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer '+self.key+
            '\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '+str(len(data))+'\r\n\r\n').encode()+data[:1])
        wait_for(lambda:client.call(client.request('status'))['claims']==1)
        process.send_signal(signal.SIGTERM)
        with self.assertRaises(subprocess.TimeoutExpired):process.wait(timeout=.5)
        self.assertEqual(client.call(client.request('status'))['claims'],1)
        connection.sendall(data[1:])
        while connection.recv(65536):pass
        out,err=process.communicate(timeout=15);self.assertEqual(process.returncode,0,err)
        self.assertIn(dict(event='stopped',scope='cpu_reference'),[json.loads(line) for line in out.splitlines()])
        self.assertEqual(client.call(client.request('status'))['claims'],0)
        self.stop(authority)

    def test_cpu_shutdown_authority_loss_is_failure_and_retains_claim(self):
        authority=self.bootstrap();client=self.client()
        fixture=reference.ReferenceCommandTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        snapshot=fixture.snapshot('glm')
        env=self.env|{'DLLM_REMOTE_QUOTA_CONFIG':str(self.config),
            'DLLM_CLIENT_KEYS':json.dumps({'alice':dict(key=self.key,max_inflight=1,max_reserved_output_tokens=8)})}
        process=subprocess.Popen([sys.executable,'-m','scripts.serve_cpu_reference','glm',str(snapshot),'--context','256','--port','0'],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.processes.append(process);ready=self.event(process)
        connection=socket.create_connection(('127.0.0.1',int(ready['url'].rsplit(':',1)[1])),timeout=5)
        self.addCleanup(connection.close)
        connection.sendall(('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer '+self.key+
            '\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{').encode())
        wait_for(lambda:client.call(client.request('status'))['claims']==1)
        self.stop(authority);process.send_signal(signal.SIGTERM);connection.sendall(b'}')
        while connection.recv(65536):pass
        out,err=process.communicate(timeout=15)
        self.assertNotEqual(process.returncode,0);self.assertNotIn('"event": "stopped"',out)
        self.assertNotIn(self.key,out+err);self.assertNotIn(self.env['DLLM_REMOTE_QUOTA_OWNER_KEY'],out+err)
        authority=self.start();client=self.client()
        self.assertEqual(client.call(client.request('status'))['claims'],1)
        self.stop(authority)


if __name__=='__main__':unittest.main()
