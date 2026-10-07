from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from dllm.http_auth import AuthPolicy,ClientKey
from dllm.shared_client_rates import SharedClientRates,SharedRateRejected,SharedRateUnavailable
from dllm.client_admission import ClientAdmission,ClientRateLimitError,ClientOutputRateLimitError
from dllm.server import handler_for
from tests.test_server import FakeRuntime


class SharedClientRateTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.root=Path(directory.name);self.path=self.root/'rates.sqlite';self.now=[.1]
        self.client=ClientKey('a',secrets.token_urlsafe(32),4,max_requests_per_minute=3,max_output_tokens_per_minute=10)

    def store(self,clients=None,path=None):
        return SharedClientRates(path or self.path,clients or [self.client],clock=lambda:self.now[0])

    def test_expiry_weighted_retry_restart_key_rotation_and_no_secret_storage(self):
        store=self.store();store.charge(self.client,'outputs',2);self.now[0]=10.2;store.charge(self.client,'outputs',8)
        self.now[0]=60.9
        with self.assertRaises(SharedRateRejected) as caught:store.charge(self.client,'outputs',3)
        self.assertEqual(caught.exception.retry_after,11)
        with self.assertRaises(SharedRateRejected):store.charge(self.client,'outputs',2)
        rotated=ClientKey('a',secrets.token_urlsafe(32),4,max_requests_per_minute=3,max_output_tokens_per_minute=10)
        reopened=self.store([rotated]);self.now[0]=61;reopened.charge(rotated,'outputs',2)
        with self.assertRaises(SharedRateRejected):reopened.charge(rotated,'outputs',1)
        self.now[0]=71;reopened.charge(rotated,'outputs',8)
        for key in (self.client.key,rotated.key):self.assertNotIn(key.encode(),self.path.read_bytes())
        self.assertEqual(self.path.stat().st_mode&0o777,0o600)

    def test_request_counts_rollback_clock_and_declared_failure_charge_are_conservative(self):
        store=self.store();self.now[0]=100.2
        for _ in range(3):store.charge(self.client,'requests',1)
        self.now[0]=90
        with self.assertRaises(SharedRateRejected) as caught:store.charge(self.client,'requests',1)
        self.assertEqual(caught.exception.retry_after,71)
        self.now[0]=161;store.charge(self.client,'requests',1)
        admission=ClientAdmission(AuthPolicy(clients=(self.client,)),rate_store=store)
        headers=['Bearer '+self.client.key]
        with self.assertRaises(RuntimeError):
            with admission.lease(headers),admission.output_lease(headers,10):raise RuntimeError('model failed')
        self.assertFalse(admission.active);self.assertFalse(admission.reserved_output)
        with self.assertRaises(ClientOutputRateLimitError):
            with admission.output_lease(headers,1):pass
        with admission.lease(headers):pass
        with self.assertRaises(ClientRateLimitError):
            with admission.lease(headers):pass

    def test_real_process_contention_restart_and_bucket_bounds(self):
        client=ClientKey('a',secrets.token_urlsafe(32),128,max_requests_per_minute=17,max_output_tokens_per_minute=101)
        self.store([client])
        code='''import json,secrets,sys
from dllm.http_auth import ClientKey
from dllm.shared_client_rates import SharedClientRates,SharedRateRejected
client=ClientKey('a',secrets.token_urlsafe(32),128,max_requests_per_minute=17,max_output_tokens_per_minute=101)
store=SharedClientRates(sys.argv[1],[client],clock=lambda:1.5)
counts=[0,0]
for _ in range(15):
 for i,(kind,amount) in enumerate((('requests',1),('outputs',7))):
  try:store.charge(client,kind,amount);counts[i]+=1
  except SharedRateRejected:pass
print(json.dumps(counts))
'''
        processes=[subprocess.Popen([sys.executable,'-c',code,str(self.path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
        try:
            results=[]
            for process in processes:
                output,error=process.communicate(timeout=15);self.assertEqual(process.returncode,0,error);results.append(json.loads(output))
            self.assertEqual([sum(row[i] for row in results) for i in (0,1)],[17,14])
        finally:
            for process in processes:
                if process.poll() is None:process.kill()
                process.communicate(timeout=5)
        restarted=self.store([client]);self.now[0]=2
        with self.assertRaises(SharedRateRejected):restarted.charge(client,'requests',1)
        with self.assertRaises(SharedRateRejected):restarted.charge(client,'outputs',7)
        path=self.root/'buckets.sqlite';large=ClientKey('b',secrets.token_urlsafe(32),1,max_output_tokens_per_minute=100000)
        store=self.store([large],path)
        for second in range(200):
            self.now[0]=second+.1;store.charge(large,'outputs',1)
        with sqlite3.connect(path) as connection:self.assertLessEqual(connection.execute('SELECT count(*) FROM buckets').fetchone()[0],61)

    def test_policy_mismatch_corruption_replacement_and_unsafe_files_fail_closed(self):
        store=self.store()
        other=ClientKey('a',secrets.token_urlsafe(32),4,max_requests_per_minute=4,max_output_tokens_per_minute=10)
        with self.assertRaises(SharedRateUnavailable):self.store([other])
        with self.assertRaises(SharedRateUnavailable):store.charge(other,'requests',1)
        original=self.path.read_bytes();self.path.unlink();self.path.write_bytes(original);self.path.chmod(0o600)
        with self.assertRaises(SharedRateUnavailable):store.charge(self.client,'requests',1)
        self.path.write_bytes(b'corrupt database')
        with self.assertRaises(SharedRateUnavailable):self.store()
        self.path.unlink();target=self.root/'target';target.write_bytes(b'unchanged')
        self.path.symlink_to(target)
        with self.assertRaises(SharedRateUnavailable):self.store()
        self.assertEqual(target.read_bytes(),b'unchanged');self.path.unlink();os.mkfifo(self.path)
        with self.assertRaises(SharedRateUnavailable):self.store()
        self.path.unlink();self.path.write_bytes(b'');self.path.chmod(0o644)
        with self.assertRaises(SharedRateUnavailable):self.store()
        self.path.unlink();self.root.chmod(0o777)
        try:
            with self.assertRaises(SharedRateUnavailable):self.store()
        finally:self.root.chmod(0o700)

    def test_storage_failure_releases_process_reservations_and_http_returns_503_before_dispatch(self):
        client=ClientKey('a',self.client.key,4,max_reserved_output_tokens=10,max_requests_per_minute=3,max_output_tokens_per_minute=10)
        auth=AuthPolicy(clients=(client,));store=self.store([client]);admission=ClientAdmission(auth,rate_store=store)
        headers=['Bearer '+client.key];calls=[]
        class Runtime(FakeRuntime):
            def generate(self,*args,**kwargs):calls.append(True);yield from super().generate(*args,**kwargs)
        with patch('dllm.client_admission.ClientAdmission',return_value=admission):handler=handler_for(Runtime(),auth)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join(3)))
        def request():
            return urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                json.dumps(dict(model='a',prompt='hello',max_tokens=2)).encode(),
                {'Content-Type':'application/json','Authorization':headers[0]}),timeout=3)
        with patch.object(store,'charge',side_effect=SharedRateUnavailable()):
            with self.assertRaises(HTTPError) as caught:request()
            self.assertEqual(caught.exception.code,503);self.assertNotIn(client.key,caught.exception.read().decode());caught.exception.close()
        self.assertFalse(calls);self.assertFalse(admission.active)
        original=store.charge
        def output_failure(client,kind,amount):
            if kind=='outputs':raise SharedRateUnavailable()
            return original(client,kind,amount)
        with patch.object(store,'charge',side_effect=output_failure):
            with self.assertRaises(HTTPError) as caught:request()
            self.assertEqual(caught.exception.code,503);caught.exception.close()
        self.assertFalse(calls);self.assertFalse(admission.active);self.assertFalse(admission.reserved_output)
        with request() as response:self.assertEqual(json.load(response)['usage']['completion_tokens'],2)

    def test_explicit_environment_opt_in_and_no_unlimited_accidental_store(self):
        auth=AuthPolicy(clients=(self.client,))
        with patch.dict('os.environ',{'DLLM_CLIENT_RATE_DB':str(self.path)}):
            first=ClientAdmission(auth);second=ClientAdmission(auth)
            for _ in range(3):
                with first.lease(['Bearer '+self.client.key]):pass
            with self.assertRaises(ClientRateLimitError):
                with second.lease(['Bearer '+self.client.key]):pass
        with patch.dict('os.environ',{},clear=True):self.assertIsNone(ClientAdmission(auth).rate_store)
        with self.assertRaises(ValueError):SharedClientRates(self.root/'unused.sqlite',[ClientKey('b',secrets.token_urlsafe(32),1)])
        self.assertFalse((self.root/'unused.sqlite').exists())
