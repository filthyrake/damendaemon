import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from dllm.http_auth import ClientKey
from dllm.shared_client_reservations import SharedClientReservations,SharedReservationRejected,SharedReservationUnavailable,Reservation

CHILD='''
import json,sys,secrets
from dllm.http_auth import ClientKey
from dllm.shared_client_reservations import SharedClientReservations,SharedReservationRejected
c=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=10)
s=SharedClientReservations(sys.argv[1],[c])
try:
 r=s.reserve(c,outputs=int(sys.argv[2]));print('reserved',flush=True)
except SharedReservationRejected:
 print('rejected',flush=True);s.close();raise SystemExit()
sys.stdin.readline()
s.release(r);s.close();print('released',flush=True)
'''

class SharedReservationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)/'claims.db'
        self.client=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=10)
        self.stores=[]
        def cleanup():
            for store in reversed(self.stores):
                for claim in list(store.claims.values()):store.release(claim)
                store.close()
        self.addCleanup(cleanup)

    def store(self,client=None):
        store=SharedClientReservations(self.path,[client or self.client]);self.stores.append(store);return store

    def child(self,outputs):
        process=subprocess.Popen([sys.executable,'-c',CHILD,str(self.path),str(outputs)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup);return process

    def test_shared_caps_resize_owner_identity_and_idempotent_release(self):
        a,b=self.store(),self.store();first=a.reserve(self.client,outputs=7);second=b.reserve(self.client,outputs=3)
        with self.assertRaises(SharedReservationRejected):a.reserve(self.client)
        with self.assertRaises(SharedReservationRejected):b.resize(second,4)
        a.resize(first,5);b.resize(second,5)
        with self.assertRaises(ValueError):b.release(first)
        with self.assertRaises(ValueError):a.release(Reservation(first.id,first.client,first.requests,object()))
        with self.assertRaises(RuntimeError):a.close()
        a.release(first);a.release(first);a.close();a.close()
        with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        self.assertFalse(a.claims)
        b.release(second)

    def test_output_only_claims_and_atomic_rejected_resize(self):
        a,b=self.store(),self.store();first=a.reserve(self.client,outputs=8,requests=0)
        second=b.reserve(self.client);third=b.reserve(self.client)
        with self.assertRaises(SharedReservationRejected) as error:a.reserve(self.client,outputs=3,requests=0)
        self.assertEqual(error.exception.kind,'outputs')
        with self.assertRaises(SharedReservationRejected):a.resize(first,11)
        extra=a.reserve(self.client,outputs=2,requests=0);a.release(extra)
        for args in (dict(requests=0),dict(outputs=True),dict(requests=True),dict(outputs=-1)):
            with self.assertRaises(ValueError):a.reserve(self.client,**args)
        a.release(first);b.release(second);b.release(third)

    def test_subprocess_contention_crash_reclaim_and_surviving_owner(self):
        a=self.store();first=a.reserve(self.client,outputs=6)
        p=self.child(4);self.assertEqual(p.stdout.readline().strip(),'reserved')
        q=self.child(1);self.assertEqual(q.stdout.readline().strip(),'rejected');self.assertEqual(q.wait(5),0)
        restarted=self.store()
        with self.assertRaises(SharedReservationRejected):restarted.reserve(self.client)
        p.kill();p.wait(5)
        replacement=restarted.reserve(self.client,outputs=4)
        with self.assertRaises(SharedReservationRejected):restarted.reserve(self.client)
        a.release(first);restarted.release(replacement)
        p=self.child(10);self.assertEqual(p.stdout.readline().strip(),'reserved')
        p.stdin.write('release\n');p.stdin.flush();self.assertEqual(p.stdout.readline().strip(),'released');self.assertEqual(p.wait(5),0)

    def test_rollback_and_uncertain_commit_retain_handles_until_retry(self):
        a,b=self.store(),self.store()
        with patch.object(a,'_commit',side_effect=sqlite3.OperationalError('injected')):
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client,outputs=10)
        failed=next(iter(a.claims.values()))
        for _ in range(3):
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        self.assertEqual(len(a.claims),1)
        other=b.reserve(self.client,outputs=10);b.release(other);a.release(failed)
        commit=a._commit
        def uncertain(connection):commit(connection);raise sqlite3.OperationalError('after commit')
        with patch.object(a,'_commit',side_effect=uncertain):
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client,outputs=10)
        claim=next(iter(a.claims.values()))
        with self.assertRaises(SharedReservationRejected):b.reserve(self.client,outputs=1)
        with patch.object(a,'_commit',side_effect=uncertain):
            with self.assertRaises(SharedReservationUnavailable):a.release(claim)
        self.assertIn(claim.id,a.claims);self.assertIsNotNone(a.descriptor)
        a.release(claim);other=b.reserve(self.client,outputs=10);b.release(other)
        with patch.object(a,'_commit',side_effect=uncertain):
            with self.assertRaises(SharedReservationUnavailable):a.close()
        self.assertFalse(a.closed);self.assertIsNotNone(a.descriptor);a.close()

    def test_policy_mismatch_lock_timeout_and_private_storage(self):
        a=self.store()
        different=ClientKey('alice',secrets.token_urlsafe(32),1,max_reserved_output_tokens=10)
        with self.assertRaises(SharedReservationUnavailable):self.store(different)
        self.assertEqual(self.path.stat().st_mode&0o777,0o600)
        connection=sqlite3.connect(self.path);connection.execute('BEGIN IMMEDIATE')
        try:
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        finally:connection.rollback();connection.close()
        for claim in list(a.claims.values()):a.release(claim)
        # Configured API keys never enter persistent storage.
        self.assertNotIn(self.client.key.encode(),self.path.read_bytes())

    def test_missing_or_replaced_owner_evidence_fails_closed(self):
        a,b=self.store(),self.store();claim=a.reserve(self.client,outputs=10)
        path=a.directory/str(a.slot);saved=path.with_suffix('.saved');path.rename(saved)
        try:
            with self.assertRaises(SharedReservationUnavailable):b.reserve(self.client)
            path.touch(mode=0o600)
            with self.assertRaises(SharedReservationUnavailable):b.reserve(self.client)
            path.unlink()
        finally:saved.rename(path)
        for failed in list(b.claims.values()):b.release(failed)
        with self.assertRaises(SharedReservationRejected):b.reserve(self.client,outputs=1)
        a.release(claim)

    def test_database_replacement_corruption_and_directory_permissions_fail_closed(self):
        a=self.store();claim=a.reserve(self.client)
        saved=self.path.with_suffix('.saved');self.path.rename(saved)
        try:
            with self.assertRaises(SharedReservationUnavailable):a.release(claim)
            with self.assertRaises(SharedReservationUnavailable):self.store()
            self.path.unlink()
        finally:saved.rename(self.path)
        os.chmod(a.directory,0o755)
        try:
            with self.assertRaises(SharedReservationUnavailable):a.release(claim)
        finally:os.chmod(a.directory,0o700)
        connection=sqlite3.connect(self.path);connection.execute('CREATE VIEW extra AS SELECT 1');connection.commit()
        try:
            with self.assertRaises(SharedReservationUnavailable):a.release(claim)
        finally:connection.execute('DROP VIEW extra');connection.commit();connection.close()
        connection=sqlite3.connect(self.path);connection.execute('PRAGMA ignore_check_constraints=ON')
        connection.execute('UPDATE claims SET requests=-1');connection.commit()
        try:
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        finally:connection.execute('UPDATE claims SET requests=1');connection.commit();connection.close()
        for pending in list(a.claims.values()):a.release(pending)

    def test_owner_and_claim_bounds_and_fork_rejection(self):
        a=self.store()
        with patch('dllm.shared_client_reservations.MAX_CLAIMS',0):
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        for claim in list(a.claims.values()):a.release(claim)
        with patch('dllm.shared_client_reservations.os.getpid',return_value=a.pid+1):
            with self.assertRaises(SharedReservationUnavailable):a.close()
            with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        for claim in list(a.claims.values()):a.release(claim)
        self.assertEqual(len(list(a.directory.iterdir())),128)
        self.assertEqual(len(self.stores),1)

    def test_fixed_owner_pool_exhaustion_and_reuse(self):
        owners=[self.store() for _ in range(128)]
        with self.assertRaises(SharedReservationUnavailable):self.store()
        owners[41].close();replacement=self.store()
        self.assertEqual(replacement.slot,41)

    def test_simultaneous_processes_cannot_oversell_capacity(self):
        # Every process creates its owner before any attempts admission.
        source=CHILD.replace("try:\n r=", "print('ready',flush=True);sys.stdin.readline()\ntry:\n r=")
        processes=[]
        for _ in range(5):
            process=subprocess.Popen([sys.executable,'-c',source,str(self.path),'4'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            processes.append(process)
            self.assertEqual(process.stdout.readline().strip(),'ready')
        try:
            for process in processes:process.stdin.write('go\n');process.stdin.flush()
            outcomes=[process.stdout.readline().strip() for process in processes]
            self.assertEqual(outcomes.count('reserved'),2);self.assertEqual(outcomes.count('rejected'),3)
        finally:
            for process in processes:
                if process.poll() is None:process.kill()
                process.communicate(timeout=5)
        owner=self.store();claim=owner.reserve(self.client,outputs=10);owner.release(claim)

    def test_symlinks_hardlinks_and_unsafe_permissions_are_rejected(self):
        real=self.path.with_suffix('.target');real.touch(mode=0o600)
        self.path.symlink_to(real)
        with self.assertRaises(SharedReservationUnavailable):self.store()
        self.path.unlink();os.link(real,self.path)
        with self.assertRaises(SharedReservationUnavailable):self.store()
        self.path.unlink();real.unlink();self.path.touch(mode=0o644)
        with self.assertRaises(SharedReservationUnavailable):self.store()
        self.path.unlink()
        a=self.store();path=a.directory/'127';saved=path.with_suffix('.saved');path.rename(saved);path.symlink_to(saved)
        try:
            with self.assertRaises(SharedReservationUnavailable):self.store()
        finally:path.unlink();saved.rename(path)

    def test_constructor_and_resize_commit_uncertainty_keep_ownership(self):
        a=self.store();claim=a.reserve(self.client,outputs=2);b=self.store()
        original=SharedClientReservations._commit
        def uncertain(store,connection):original(store,connection);raise sqlite3.OperationalError('committed')
        with patch.object(SharedClientReservations,'_commit',uncertain):
            with self.assertRaises(SharedReservationUnavailable) as error:SharedClientReservations(self.path,[self.client])
        retained=error.exception.store;self.assertIsNotNone(retained);self.stores.append(retained)
        self.assertIsNotNone(retained.descriptor);retained.close()
        with patch.object(a,'_commit',lambda connection:uncertain(a,connection)):
            with self.assertRaises(SharedReservationUnavailable):a.resize(claim,10)
        with self.assertRaises(SharedReservationUnavailable):a.reserve(self.client)
        with self.assertRaises(SharedReservationRejected):b.reserve(self.client,outputs=1)
        a.resize(claim,2);other=b.reserve(self.client,outputs=8);b.release(other);a.release(claim)
