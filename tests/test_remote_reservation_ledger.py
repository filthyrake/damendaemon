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
from dllm.remote_reservation_ledger import RemoteReservationLedger,RemoteLedgerUnavailable,RemoteOwnerDenied,RemoteSequenceError


def reserve(outputs=0,requests=1,client='alice'):return dict(op='reserve',client=client,requests=requests,outputs=outputs)


class RemoteReservationLedgerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)/'remote.sqlite'
        self.client=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=10)
        self.bob=ClientKey('bob',secrets.token_urlsafe(32),1,max_reserved_output_tokens=5)
        self.ledger=RemoteReservationLedger(self.path,[self.client,self.bob],create=True);self.addCleanup(self.ledger.close)
        self.owner,self.key=secrets.token_hex(16),secrets.token_bytes(32)
        self.ledger.register(self.owner,self.key,['alice'])

    def reopen(self):
        ledger=RemoteReservationLedger(self.path,[self.client,self.bob]);self.addCleanup(ledger.close);return ledger

    def run_command(self,seq,command,ledger=None):return (ledger or self.ledger).execute(self.owner,self.key,seq,command)

    def test_caps_resizing_idempotency_retirement_and_owner_isolation(self):
        a=self.run_command(1,reserve(8));self.assertTrue(a['ok']);self.assertEqual(a,self.run_command(1,reserve(8)))
        with self.assertRaises(RemoteSequenceError):self.run_command(1,reserve(7))
        with self.assertRaises(RemoteSequenceError):self.run_command(3,reserve())
        self.assertEqual(self.run_command(2,reserve(3))['code'],'output_limit')
        self.assertEqual(self.run_command(3,dict(op='resize',claim=a['claim'],outputs=6))['code'],'resized')
        b=self.run_command(4,reserve(4));self.assertTrue(b['ok'])
        self.assertEqual(self.run_command(5,reserve())['code'],'request_limit')
        self.assertEqual(self.run_command(6,dict(op='retire'))['code'],'owner_busy')
        owner,key=secrets.token_hex(16),secrets.token_bytes(32);self.ledger.register(owner,key,['alice'])
        with self.assertRaises(RemoteOwnerDenied):self.ledger.execute(owner,key,1,dict(op='release',claim=a['claim']))
        with self.assertRaises(RemoteOwnerDenied):self.run_command(7,reserve(client='bob'))
        with self.assertRaises(RemoteOwnerDenied):self.ledger.status(self.owner,secrets.token_bytes(32))
        self.run_command(7,dict(op='release',claim=a['claim']));self.run_command(8,dict(op='release',claim=b['claim']))
        done=self.run_command(9,dict(op='retire'));self.assertEqual(done,self.run_command(9,dict(op='retire')))
        with self.assertRaises(RemoteOwnerDenied):self.run_command(10,reserve())
        with self.assertRaises(RemoteOwnerDenied):self.ledger.register(self.owner,self.key,['alice'])

    def test_authority_restart_and_close_never_reclaim_remote_claims(self):
        result=self.run_command(1,reserve(10));self.ledger.close();ledger=self.reopen()
        self.assertEqual(ledger.status(self.owner,self.key)['claims'],1)
        self.assertEqual(self.run_command(1,reserve(10),ledger),result)
        owner,key=secrets.token_hex(16),secrets.token_bytes(32);ledger.register(owner,key,['alice'])
        self.assertEqual(ledger.execute(owner,key,1,reserve(1))['code'],'output_limit')
        self.run_command(2,dict(op='release',claim=result['claim']),ledger)
        # Rejection receipts stay rejected even after capacity becomes available.
        self.assertEqual(ledger.execute(owner,key,1,reserve(1))['code'],'output_limit')
        self.assertTrue(ledger.execute(owner,key,2,reserve(10))['ok'])

    def test_response_loss_before_and_after_commit_retries_exactly_once(self):
        commit=self.ledger._commit
        with patch.object(self.ledger,'_commit',side_effect=sqlite3.OperationalError('before')):
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(1,reserve(6))
        self.assertEqual(self.ledger.status(self.owner,self.key)['sequence'],0)
        def uncertain(connection):commit(connection);raise sqlite3.OperationalError('after')
        with patch.object(self.ledger,'_commit',uncertain):
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(1,reserve(6))
        claim=self.run_command(1,reserve(6))['claim'];self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],1)
        with patch.object(self.ledger,'_commit',uncertain):
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(2,dict(op='resize',claim=claim,outputs=10))
        self.assertTrue(self.run_command(2,dict(op='resize',claim=claim,outputs=10))['ok'])
        with patch.object(self.ledger,'_commit',uncertain):
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(3,dict(op='release',claim=claim))
        self.assertTrue(self.run_command(3,dict(op='release',claim=claim))['ok']);self.assertEqual(self.ledger.status(self.owner,self.key)['claims'],0)

    def test_registration_retry_and_plaintext_credentials_never_persist(self):
        self.assertEqual(self.ledger.register(self.owner,self.key,['alice'])['owner'],self.owner)
        with self.assertRaises(RemoteOwnerDenied):self.ledger.register(self.owner,secrets.token_bytes(32),['alice'])
        with self.assertRaises(RemoteOwnerDenied):self.ledger.register(self.owner,self.key,['bob'])
        self.run_command(1,reserve())
        data=self.path.read_bytes()
        self.assertNotIn(self.key,data);self.assertNotIn(self.key.hex().encode(),data);self.assertNotIn(self.client.key.encode(),data)
        owner,key=secrets.token_hex(16),secrets.token_bytes(32);commit=self.ledger._commit
        def uncertain(connection):commit(connection);raise sqlite3.OperationalError('response lost')
        with patch.object(self.ledger,'_commit',uncertain):
            with self.assertRaises(RemoteLedgerUnavailable):self.ledger.register(owner,key,['alice'])
        self.assertEqual(self.ledger.register(owner,key,['alice'])['owner'],owner)

    def test_process_contention_and_process_exit_leave_capacity_charged(self):
        code='''import json,secrets,sys
from dllm.http_auth import ClientKey
from dllm.remote_reservation_ledger import RemoteReservationLedger
client=ClientKey('alice',secrets.token_urlsafe(32),2,max_reserved_output_tokens=10)
ledger=RemoteReservationLedger(sys.argv[1],[client])
owner,key=secrets.token_hex(16),secrets.token_bytes(32)
ledger.register(owner,key,['alice'])
result=ledger.execute(owner,key,1,dict(op='reserve',client='alice',requests=1,outputs=5))
print(json.dumps(result),flush=True)
'''
        processes=[subprocess.Popen([sys.executable,'-c',code,str(self.path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(6)]
        try:
            results=[]
            for process in processes:
                output,error=process.communicate(timeout=10);self.assertEqual(process.returncode,0,error);results.append(json.loads(output))
            self.assertEqual(sum(result['ok'] for result in results),2)
        finally:
            for process in processes:
                if process.poll() is None:process.kill()
                process.communicate(timeout=5)
        ledger=self.reopen();self.assertEqual(self.run_command(1,reserve(),ledger)['code'],'request_limit')

    def test_storage_replacement_missing_anchor_and_corruption_fail_closed(self):
        claim=self.run_command(1,reserve())['claim'];saved=self.path.with_suffix('.saved');self.path.rename(saved)
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.reopen()
            with self.assertRaises(RemoteLedgerUnavailable):RemoteReservationLedger(self.path,[self.client],create=True)
            self.path.unlink()
        finally:saved.rename(self.path)
        anchor=self.ledger.anchor;backup=anchor.with_suffix('.saved');anchor.rename(backup)
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(2,dict(op='release',claim=claim))
            with self.assertRaises(RemoteLedgerUnavailable):self.reopen()
        finally:backup.rename(anchor)
        connection=sqlite3.connect(self.path);connection.execute('CREATE VIEW bad AS SELECT 1');connection.commit()
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.ledger.status(self.owner,self.key)
        finally:connection.execute('DROP VIEW bad');connection.commit();connection.close()
        self.assertEqual(self.reopen().status(self.owner,self.key)['claims'],1)
        self.run_command(2,dict(op='release',claim=claim))

    def test_bounds_unsafe_storage_and_lock_contention(self):
        with patch('dllm.remote_reservation_ledger.MAX_OWNERS',1):
            with self.assertRaises(RemoteLedgerUnavailable):self.ledger.register(secrets.token_hex(16),secrets.token_bytes(32),['alice'])
        with patch('dllm.remote_reservation_ledger.MAX_CLAIMS',0):self.assertEqual(self.run_command(1,reserve())['code'],'claim_limit')
        connection=sqlite3.connect(self.path);connection.execute('BEGIN IMMEDIATE')
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.run_command(2,reserve())
        finally:connection.rollback();connection.close()
        self.assertTrue(self.run_command(2,reserve())['ok'])
        os.chmod(self.path,0o644)
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.reopen()
        finally:os.chmod(self.path,0o600)
        different=ClientKey('alice',secrets.token_urlsafe(32),1,max_reserved_output_tokens=10)
        with self.assertRaises(RemoteLedgerUnavailable):RemoteReservationLedger(self.path,[different])
        with self.assertRaises(RemoteLedgerUnavailable):RemoteReservationLedger(self.path,[self.client],create=True)

    def test_bounded_owner_claim_pages_and_strict_commands(self):
        for sequence in range(1,6):self.run_command(sequence,reserve(1,requests=0))
        first=self.ledger.list_claims(self.owner,self.key,limit=2);second=self.ledger.list_claims(self.owner,self.key,limit=2,after=first['after'])
        third=self.ledger.list_claims(self.owner,self.key,limit=2,after=second['after'])
        self.assertEqual([len(page['claims']) for page in (first,second,third)],[2,2,1]);self.assertIsNone(third['after'])
        for command in (reserve(True),reserve(requests=True),dict(op='reserve'),reserve()|dict(secret='unexpected'),reserve(client=[])):
            with self.assertRaises((ValueError,RemoteOwnerDenied)):self.run_command(6,command)
        self.assertEqual(self.ledger.status(self.owner,self.key)['sequence'],5)
        claim=first['claims'][0]['claim']
        self.assertEqual(self.run_command(6,dict(op='resize',claim=claim,outputs=0))['code'],'empty_claim')
        for limit in (True,0,129):
            with self.assertRaises(ValueError):self.ledger.list_claims(self.owner,self.key,limit=limit)

    def test_retired_identities_are_bounded_and_never_resurrected(self):
        self.run_command(1,dict(op='retire'))
        for _ in range(127):
            owner,key=secrets.token_hex(16),secrets.token_bytes(32)
            self.ledger.register(owner,key,['alice'])
            self.assertTrue(self.ledger.execute(owner,key,1,dict(op='retire'))['ok'])
        with self.assertRaises(RemoteLedgerUnavailable):self.ledger.register(secrets.token_hex(16),secrets.token_bytes(32),['alice'])
        with self.assertRaises(RemoteOwnerDenied):self.reopen().register(self.owner,self.key,['alice'])

    def test_anchor_changes_invalid_receipt_and_storage_bounds_are_rejected(self):
        self.run_command(1,reserve())
        anchor=self.ledger.anchor;original=anchor.read_bytes();anchor.write_bytes(b'{}')
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.ledger.status(self.owner,self.key)
        finally:anchor.write_bytes(original)
        connection=sqlite3.connect(self.path)
        saved=connection.execute('SELECT receipt FROM owners WHERE id=?',(self.owner,)).fetchone()[0]
        connection.execute('UPDATE owners SET receipt=? WHERE id=?',('{}',self.owner));connection.commit()
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.reopen()
        finally:connection.execute('UPDATE owners SET receipt=? WHERE id=?',(saved,self.owner));connection.commit();connection.close()
        connection=sqlite3.connect(self.path);connection.execute('PRAGMA ignore_check_constraints=ON')
        connection.execute('UPDATE claims SET requests=-1');connection.commit()
        try:
            with self.assertRaises(RemoteLedgerUnavailable):self.ledger.status(self.owner,self.key)
        finally:connection.execute('UPDATE claims SET requests=1');connection.commit();connection.close()
        unused=self.path.with_name('absent.sqlite')
        with self.assertRaises(RemoteLedgerUnavailable):RemoteReservationLedger(unused,[self.client])
        self.assertFalse(unused.exists())
        alias=self.path.with_name('alias.sqlite');alias.symlink_to(self.path)
        with self.assertRaises(RemoteLedgerUnavailable):RemoteReservationLedger(alias,[self.client])
