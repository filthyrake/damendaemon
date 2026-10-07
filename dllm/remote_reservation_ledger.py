"""Durable remote-owner quotas. Authority restart never reclaims remote work.

This is a local accounting library, not a network service or workload fence.
Bootstrap is explicit. Stable owners retain their last operation receipt so a
lost response can be retried without charging twice. No TTL or disconnect path
returns capacity; authenticated release is required.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading

APP_ID=0x444C5252
MAX_BYTES=32*1024**2
MAX_OWNERS=128
MAX_CLAIMS=32768
MAX_SEQUENCE=2**63-1
SCHEMA=(
    'CREATE TABLE identity(id TEXT PRIMARY KEY)',
    'CREATE TABLE clients(name TEXT PRIMARY KEY, requests INTEGER NOT NULL CHECK(requests BETWEEN 1 AND 128), outputs INTEGER CHECK(outputs BETWEEN 1 AND 2147483647))',
    'CREATE TABLE owners(id TEXT PRIMARY KEY, credential TEXT NOT NULL, allowed TEXT NOT NULL, retired INTEGER NOT NULL CHECK(retired IN (0,1)), sequence INTEGER NOT NULL CHECK(sequence>=0), digest TEXT, receipt TEXT CHECK(length(receipt)<=1024))',
    'CREATE TABLE claims(id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES owners(id), client TEXT NOT NULL REFERENCES clients(name), requests INTEGER NOT NULL CHECK(requests IN (0,1)), outputs INTEGER NOT NULL CHECK(outputs BETWEEN 0 AND 2147483647), CHECK(requests+outputs>0))',
)
INDEXES=('CREATE INDEX claims_owner ON claims(owner)','CREATE INDEX claims_client ON claims(client)')
CODES={'reserved','resized','released','retired','request_limit','output_limit','unknown_claim','owner_busy','claim_limit','empty_claim'}


class RemoteLedgerUnavailable(RuntimeError):
    def __init__(self):super().__init__('remote reservation ledger unavailable; retry the same operation')


class RemoteOwnerDenied(Exception):
    def __init__(self):super().__init__('remote reservation owner denied')


class RemoteSequenceError(Exception):
    def __init__(self):super().__init__('remote reservation operation sequence conflict')


def _hex(value,size):return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(size)+'}',value) is not None


def _encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


class RemoteReservationLedger:
    """Owners serialize commands using increasing sequence numbers from one.

    Retry only the last command with its original sequence and exact payload.
    Rejections have durable receipts too. Earlier/out-of-order or changed retries
    cannot mutate state. Registered identities, including retired tombstones, are
    bounded to 128; use stable deployment identities, not one owner per request.
    ``close`` only closes this local handle and never retires a remote owner.
    """
    def __init__(self,path,clients,*,create=False):
        from .http_auth import ClientKey
        if type(create) is not bool:raise ValueError('create must be boolean')
        if (not isinstance(clients,(list,tuple)) or not 1<=len(clients)<=128
                or any(not isinstance(c,ClientKey) for c in clients)
                or len({c.name for c in clients})!=len(clients)):
            raise ValueError('remote ledger requires bounded named client policies')
        if not hasattr(os,'O_NOFOLLOW') or not hasattr(os,'getuid'):raise ValueError('remote ledger requires private POSIX storage')
        raw=Path(path).expanduser()
        if not raw.is_absolute():raise ValueError('remote ledger path must be absolute')
        self.path=raw.parent.resolve(strict=True)/raw.name;self.anchor=self.path.with_name(self.path.name+'.identity')
        self.policies={c.name:(c.max_inflight,c.max_reserved_output_tokens) for c in clients}
        self.mutex=threading.RLock();self.closed=False;self.file_identity=None;self.anchor_identity=None;self.anchor_bytes=None
        try:
            self._parent()
            flags=os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK
            fd=os.open(self.path,flags|(os.O_CREAT|os.O_EXCL if create else 0),0o600)
            try:self.file_identity=self._file(os.fstat(fd),MAX_BYTES)
            finally:os.close(fd)
            if create:
                self.identity=secrets.token_hex(16)
                self.anchor_bytes=_encode(dict(device=self.file_identity[0],inode=self.file_identity[1],id=self.identity)).encode()
                fd=os.open(self.anchor,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                try:
                    self.anchor_identity=self._file(os.fstat(fd),256)
                    if os.write(fd,self.anchor_bytes)!=len(self.anchor_bytes):raise RemoteLedgerUnavailable()
                    os.fsync(fd)
                finally:os.close(fd)
                fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            else:
                self._anchor()
                data=json.loads(self.anchor_bytes)
                if (set(data)!={'device','inode','id'} or (data['device'],data['inode'])!=self.file_identity
                        or not _hex(data['id'],32)):raise RemoteLedgerUnavailable()
                self.identity=data['id']
            with self._transaction(initialize=create) as connection:
                if create:
                    for sql in (*SCHEMA,*INDEXES):connection.execute(sql)
                    connection.execute('INSERT INTO identity VALUES(?)',(self.identity,))
                    connection.execute(f'PRAGMA application_id={APP_ID}');connection.execute('PRAGMA user_version=1')
                if connection.execute('PRAGMA quick_check').fetchone()!=('ok',):raise RemoteLedgerUnavailable()
                for name,policy in self.policies.items():
                    row=connection.execute('SELECT requests,outputs FROM clients WHERE name=?',(name,)).fetchone()
                    if row is None:
                        if connection.execute('SELECT count(*) FROM clients').fetchone()[0]>=128:raise RemoteLedgerUnavailable()
                        connection.execute('INSERT INTO clients VALUES(?,?,?)',(name,*policy))
                    elif row!=policy:raise RemoteLedgerUnavailable()
        except RemoteLedgerUnavailable:raise
        except (OSError,sqlite3.Error,ValueError,TypeError,OverflowError):raise RemoteLedgerUnavailable() from None

    def _parent(self):
        info=self.path.parent.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o022:raise RemoteLedgerUnavailable()

    @staticmethod
    def _file(info,maximum):
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077
                or info.st_nlink!=1 or info.st_size>maximum):raise RemoteLedgerUnavailable()
        return info.st_dev,info.st_ino

    def _anchor(self):
        fd=os.open(self.anchor,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            identity=self._file(os.fstat(fd),256);data=os.read(fd,257)
            if len(data)>256 or (self.anchor_identity is not None and identity!=self.anchor_identity):raise RemoteLedgerUnavailable()
            if self.anchor_bytes is not None and data!=self.anchor_bytes:raise RemoteLedgerUnavailable()
            self.anchor_identity,self.anchor_bytes=identity,data
        finally:os.close(fd)

    def _validate(self,connection):
        if {row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE type='table'")}!=set(SCHEMA):raise RemoteLedgerUnavailable()
        if {row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")}!=set(INDEXES):raise RemoteLedgerUnavailable()
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger','view') LIMIT 1").fetchone():raise RemoteLedgerUnavailable()
        if connection.execute('SELECT id FROM identity').fetchall()!=[(self.identity,)]:raise RemoteLedgerUnavailable()
        if (connection.execute('SELECT count(*) FROM owners').fetchone()[0]>MAX_OWNERS
                or connection.execute('SELECT count(*) FROM clients').fetchone()[0]>128
                or connection.execute('SELECT count(*) FROM claims').fetchone()[0]>MAX_CLAIMS
                or connection.execute('PRAGMA foreign_key_check').fetchone()):raise RemoteLedgerUnavailable()
        clients=dict((name,(requests,outputs)) for name,requests,outputs in connection.execute('SELECT * FROM clients'))
        for name,(requests,outputs) in clients.items():
            if (not isinstance(name,str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',name) is None
                    or type(requests) is not int or not 1<=requests<=128
                    or (outputs is not None and (type(outputs) is not int or not 1<=outputs<=2**31-1))):raise RemoteLedgerUnavailable()
        for name,policy in self.policies.items():
            if name in clients and clients[name]!=policy:raise RemoteLedgerUnavailable()
        owner_policies={}
        for owner,credential,allowed,retired,sequence,digest,receipt in connection.execute('SELECT * FROM owners'):
            names=json.loads(allowed)
            if (not _hex(owner,32) or not _hex(credential,64) or not isinstance(names,list)
                    or not 1<=len(names)<=128 or any(not isinstance(name,str) or name not in clients for name in names)
                    or names!=sorted(set(names)) or type(retired) is not int or retired not in (0,1)
                    or type(sequence) is not int or not 0<=sequence<=MAX_SEQUENCE):raise RemoteLedgerUnavailable()
            owner_policies[owner]=(names,sequence)
            if sequence==0:
                if digest is not None or receipt is not None or retired:raise RemoteLedgerUnavailable()
            else:
                if not _hex(digest,64) or not isinstance(receipt,str) or len(receipt)>1024:raise RemoteLedgerUnavailable()
                result=json.loads(receipt)
                if (not isinstance(result,dict) or set(result)-{'sequence','ok','code','claim'}
                        or type(result.get('sequence')) is not int or result.get('sequence')!=sequence or type(result.get('ok')) is not bool
                        or result.get('code') not in CODES or ('claim' in result and not self._claim_id(result['claim'],owner))):raise RemoteLedgerUnavailable()
                success=result['code'] in ('reserved','resized','released','retired')
                if result['ok']!=success or ('claim' in result)!=(result['code'] in ('reserved','resized','released')):raise RemoteLedgerUnavailable()
            if retired and connection.execute('SELECT 1 FROM claims WHERE owner=? LIMIT 1',(owner,)).fetchone():raise RemoteLedgerUnavailable()
        for name,requests,outputs in connection.execute('SELECT client,sum(requests),sum(outputs) FROM claims GROUP BY client'):
            caps=clients[name]
            if type(requests) is not int or type(outputs) is not int or requests>caps[0] or (caps[1] is not None and outputs>caps[1]):raise RemoteLedgerUnavailable()
        for claim,owner,client,requests,outputs in connection.execute('SELECT * FROM claims'):
            if (not self._claim_id(claim,owner) or type(requests) is not int or type(outputs) is not int
                    or requests not in (0,1) or not 0<=outputs<=2**31-1 or requests+outputs==0
                    or client not in owner_policies[owner][0] or int(claim.split(':')[1])>owner_policies[owner][1]):raise RemoteLedgerUnavailable()

    @contextmanager
    def _transaction(self,*,initialize=False):
        connection=None;fd=None
        with self.mutex:
            if self.closed:raise RemoteLedgerUnavailable()
            try:
                self._parent();self._anchor()
                fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                if self._file(os.fstat(fd),MAX_BYTES)!=self.file_identity:raise RemoteLedgerUnavailable()
                connection=sqlite3.connect(str(self.path),timeout=1,isolation_level=None)
                connection.execute('PRAGMA trusted_schema=OFF');connection.execute('PRAGMA foreign_keys=ON');connection.execute('PRAGMA synchronous=FULL')
                if connection.execute('PRAGMA journal_mode').fetchone()[0]!='delete' or connection.execute('PRAGMA page_size').fetchone()[0]!=4096:raise RemoteLedgerUnavailable()
                connection.execute('PRAGMA max_page_count=8192');connection.execute('BEGIN IMMEDIATE')
                app=connection.execute('PRAGMA application_id').fetchone()[0];version=connection.execute('PRAGMA user_version').fetchone()[0]
                if initialize:
                    if app or version or connection.execute('SELECT count(*) FROM sqlite_master').fetchone()[0]:raise RemoteLedgerUnavailable()
                elif app!=APP_ID or version!=1:raise RemoteLedgerUnavailable()
                else:self._validate(connection)
                yield connection
                self._anchor()
                if self._file(os.stat(self.path,follow_symlinks=False),MAX_BYTES)!=self.file_identity:raise RemoteLedgerUnavailable()
                self._commit(connection)
            except (RemoteLedgerUnavailable,RemoteOwnerDenied,RemoteSequenceError):raise
            except (OSError,sqlite3.Error,ValueError,TypeError,OverflowError):raise RemoteLedgerUnavailable() from None
            finally:
                if connection is not None:connection.close()
                if fd is not None:os.close(fd)

    def _commit(self,connection):connection.commit()

    @staticmethod
    def _credentials(owner,credential):
        if not _hex(owner,32) or not isinstance(credential,bytes) or len(credential)!=32:raise RemoteOwnerDenied()
        return hashlib.sha256(credential).hexdigest()

    def register(self,owner,credential,clients):
        """Privileged local registration; a future transport must authorize it."""
        digest=self._credentials(owner,credential)
        if (not isinstance(clients,(list,tuple)) or not 1<=len(clients)<=128
                or any(not isinstance(name,str) or name not in self.policies for name in clients)
                or len(set(clients))!=len(clients)):raise ValueError('invalid remote owner client policy')
        allowed=_encode(sorted(clients))
        with self._transaction() as connection:
            row=connection.execute('SELECT credential,allowed,retired FROM owners WHERE id=?',(owner,)).fetchone()
            if row is not None:
                if not secrets.compare_digest(row[0],digest) or row[1]!=allowed or row[2]:raise RemoteOwnerDenied()
            else:
                if connection.execute('SELECT count(*) FROM owners').fetchone()[0]>=MAX_OWNERS:raise RemoteLedgerUnavailable()
                connection.execute('INSERT INTO owners VALUES(?,?,?,0,0,NULL,NULL)',(owner,digest,allowed))
        return dict(owner=owner,ledger=self.identity)

    def _owner(self,connection,owner,credential):
        digest=self._credentials(owner,credential)
        row=connection.execute('SELECT credential,allowed,retired,sequence,digest,receipt FROM owners WHERE id=?',(owner,)).fetchone()
        if row is None or not secrets.compare_digest(row[0],digest):raise RemoteOwnerDenied()
        return row

    @staticmethod
    def _claim_id(claim,owner):
        if not isinstance(claim,str) or not claim.startswith(owner+':'):return False
        suffix=claim[len(owner)+1:]
        return bool(re.fullmatch('[1-9][0-9]{0,18}',suffix)) and int(suffix)<=MAX_SEQUENCE

    def _command(self,owner,command):
        if not isinstance(command,dict):raise ValueError('invalid remote reservation command')
        kind=command.get('op');fields={'reserve':{'op','client','requests','outputs'},'resize':{'op','claim','outputs'},'release':{'op','claim'},'retire':{'op'}}
        if not isinstance(kind,str) or kind not in fields or set(command)!=fields[kind]:raise ValueError('invalid remote reservation command')
        if kind=='reserve':
            if not isinstance(command['client'],str) or command['client'] not in self.policies:raise ValueError('unknown remote client policy')
            if type(command['requests']) is not int or command['requests'] not in (0,1):raise ValueError('invalid remote request count')
        if kind in ('reserve','resize'):
            if type(command['outputs']) is not int or not 0<=command['outputs']<=2**31-1:raise ValueError('invalid remote output count')
            if kind=='reserve' and command['requests']+command['outputs']==0:raise ValueError('empty reservation')
        if kind in ('resize','release') and not self._claim_id(command['claim'],owner):raise RemoteOwnerDenied()
        return _encode(command)

    def execute(self,owner,credential,sequence,command,*,operation_id=None):
        self._credentials(owner,credential)
        if type(sequence) is not int or not 1<=sequence<=MAX_SEQUENCE:raise RemoteSequenceError()
        if operation_id is not None and not _hex(operation_id,32):raise ValueError('invalid remote operation identity')
        payload=self._command(owner,command);command=json.loads(payload)
        identity=payload if operation_id is None else _encode(dict(operation_id=operation_id,command=command))
        digest=hashlib.sha256(identity.encode()).hexdigest()
        with self._transaction() as connection:
            _,allowed,retired,current,previous,receipt=self._owner(connection,owner,credential)
            if sequence==current:
                if previous!=digest:raise RemoteSequenceError()
                return json.loads(receipt)
            if retired:raise RemoteOwnerDenied()
            if sequence!=current+1:raise RemoteSequenceError()
            result=self._apply(connection,owner,json.loads(allowed),sequence,command)
            result=dict(sequence=sequence,**result)
            connection.execute('UPDATE owners SET sequence=?,digest=?,receipt=? WHERE id=?',(sequence,digest,_encode(result),owner))
        return result

    def _apply(self,connection,owner,allowed,sequence,command):
        op=command['op']
        if op=='retire':
            if connection.execute('SELECT 1 FROM claims WHERE owner=? LIMIT 1',(owner,)).fetchone():return dict(ok=False,code='owner_busy')
            connection.execute('UPDATE owners SET retired=1 WHERE id=?',(owner,));return dict(ok=True,code='retired')
        claim=owner+':'+str(sequence) if op=='reserve' else command['claim']
        row=None if op=='reserve' else connection.execute('SELECT client,requests,outputs FROM claims WHERE id=? AND owner=?',(claim,owner)).fetchone()
        if op=='release':
            connection.execute('DELETE FROM claims WHERE id=? AND owner=?',(claim,owner));return dict(ok=True,code='released',claim=claim)
        if op=='resize' and row is None:return dict(ok=False,code='unknown_claim')
        client,requests=(command['client'],command['requests']) if op=='reserve' else row[:2]
        if client not in allowed:raise RemoteOwnerDenied()
        outputs=command['outputs']
        if requests+outputs==0:return dict(ok=False,code='empty_claim')
        caps=connection.execute('SELECT requests,outputs FROM clients WHERE name=?',(client,)).fetchone()
        used=connection.execute('SELECT coalesce(sum(requests),0),coalesce(sum(outputs),0) FROM claims WHERE client=? AND id!=?',(client,claim)).fetchone()
        if requests>caps[0]-used[0]:return dict(ok=False,code='request_limit')
        if caps[1] is not None and outputs>caps[1]-used[1]:return dict(ok=False,code='output_limit')
        if op=='reserve':
            if connection.execute('SELECT count(*) FROM claims').fetchone()[0]>=MAX_CLAIMS:return dict(ok=False,code='claim_limit')
            connection.execute('INSERT INTO claims VALUES(?,?,?,?,?)',(claim,owner,client,requests,outputs))
        else:connection.execute('UPDATE claims SET outputs=? WHERE id=?',(outputs,claim))
        return dict(ok=True,code='reserved' if op=='reserve' else 'resized',claim=claim)

    def status(self,owner,credential):
        with self._transaction() as connection:
            _,_,retired,sequence,_,receipt=self._owner(connection,owner,credential)
            count=connection.execute('SELECT count(*) FROM claims WHERE owner=?',(owner,)).fetchone()[0]
        return dict(ledger=self.identity,sequence=sequence,retired=bool(retired),claims=count,receipt=None if receipt is None else json.loads(receipt))

    def list_claims(self,owner,credential,*,after=None,limit=128):
        if type(limit) is not int or not 1<=limit<=128:raise ValueError('invalid claim page limit')
        if after is not None and not self._claim_id(after,owner):raise RemoteOwnerDenied()
        with self._transaction() as connection:
            self._owner(connection,owner,credential)
            rows=connection.execute('SELECT id,client,requests,outputs FROM claims WHERE owner=? AND id>? ORDER BY id LIMIT ?',
                (owner,after or '',limit+1)).fetchall()
        return dict(claims=[dict(zip(('claim','client','requests','outputs'),row)) for row in rows[:limit]],
                    after=rows[limit-1][0] if len(rows)>limit else None)

    def close(self):
        with self.mutex:self.closed=True
