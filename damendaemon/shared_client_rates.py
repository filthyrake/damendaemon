"""Explicit single-host transactional rate buckets; no secrets or request content."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sqlite3
import stat
import time

APP_ID=0x444C5254
MAX_BYTES=32*1024**2


class SharedRateUnavailable(RuntimeError):
    def __init__(self):super().__init__('shared client rate accounting unavailable')


class SharedRateRejected(Exception):
    def __init__(self,retry_after):self.retry_after=retry_after


class SharedClientRates:
    """Share named-client rolling allowances across local processes/restarts.

    An explicit private regular SQLite file and non-writable-by-others parent
    are required. Each operation uses a short independent connection/transaction;
    lock waiting is bounded to one second. Local filesystem calls are not hard
    realtime. Network filesystems and hostile privileged writers are unsupported.
    """
    def __init__(self,path,clients,*,clock=None):
        from .http_auth import ClientKey
        if (not isinstance(clients,(list,tuple)) or not 1<=len(clients)<=128
                or any(not isinstance(c,ClientKey) for c in clients)
                or len({c.name for c in clients})!=len(clients)
                or not any(c.max_requests_per_minute is not None or c.max_output_tokens_per_minute is not None for c in clients)):
            raise ValueError('shared client rates require bounded named clients with a rate policy')
        raw=Path(path).expanduser()
        if not raw.is_absolute():raise ValueError('shared client rate path must be absolute')
        self.path=raw.parent.resolve(strict=True)/raw.name
        self.clock=time.time if clock is None else clock
        self.policies={c.name:(c.max_requests_per_minute,c.max_output_tokens_per_minute) for c in clients}
        self.file_identity=None
        try:
            self._parent()
            descriptor=os.open(self.path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
            try:self.file_identity=self._file(os.fstat(descriptor))
            finally:os.close(descriptor)
            with self._transaction(initialize=True) as connection:
                app=connection.execute('PRAGMA application_id').fetchone()[0]
                version=connection.execute('PRAGMA user_version').fetchone()[0]
                tables=connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if app==0 and version==0 and not tables:
                    connection.execute('CREATE TABLE clients(name TEXT PRIMARY KEY, request_limit INTEGER, output_limit INTEGER, last_seen REAL NOT NULL)')
                    connection.execute('CREATE TABLE buckets(client TEXT NOT NULL REFERENCES clients(name), expires INTEGER NOT NULL, requests INTEGER NOT NULL, outputs INTEGER NOT NULL, PRIMARY KEY(client,expires))')
                    connection.execute(f'PRAGMA application_id={APP_ID}')
                    connection.execute('PRAGMA user_version=1')
                elif app!=APP_ID or version!=1 or {row[0] for row in tables}!={'clients','buckets'}:
                    raise SharedRateUnavailable()
                if connection.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('trigger','view')").fetchone()[0]:
                    raise SharedRateUnavailable()
                count=connection.execute('SELECT count(*) FROM clients').fetchone()[0]
                if count>128:raise SharedRateUnavailable()
                for name,policy in self.policies.items():
                    previous=connection.execute('SELECT request_limit,output_limit FROM clients WHERE name=?',(name,)).fetchone()
                    if previous is None:
                        if count>=128:raise SharedRateUnavailable()
                        connection.execute('INSERT INTO clients VALUES(?,?,?,0)',(name,*policy));count+=1
                    elif previous!=policy:raise SharedRateUnavailable()
        except SharedRateUnavailable:raise
        except (OSError,sqlite3.Error,ValueError,TypeError,OverflowError):raise SharedRateUnavailable() from None

    def _parent(self):
        info=self.path.parent.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o022:
            raise SharedRateUnavailable()

    def _file(self,info):
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077
                or info.st_nlink!=1 or info.st_size>MAX_BYTES):raise SharedRateUnavailable()
        identity=info.st_dev,info.st_ino
        if self.file_identity is not None and identity!=self.file_identity:raise SharedRateUnavailable()
        return identity

    @contextmanager
    def _transaction(self,*,initialize=False):
        connection=None;descriptor=None
        try:
            self._parent()
            descriptor=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            self._file(os.fstat(descriptor))
            connection=sqlite3.connect(str(self.path),timeout=1,isolation_level=None)
            connection.execute('PRAGMA trusted_schema=OFF')
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA synchronous=FULL')
            app=connection.execute('PRAGMA application_id').fetchone()[0]
            version=connection.execute('PRAGMA user_version').fetchone()[0]
            if app!=APP_ID or version!=1:
                if (not initialize or app!=0 or version!=0
                        or connection.execute('SELECT count(*) FROM sqlite_master').fetchone()[0]):raise SharedRateUnavailable()
            if connection.execute('PRAGMA journal_mode').fetchone()[0]!='delete':raise SharedRateUnavailable()
            if connection.execute('PRAGMA page_size').fetchone()[0]!=4096:raise SharedRateUnavailable()
            connection.execute('PRAGMA max_page_count=8192')
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            self._file(os.stat(self.path,follow_symlinks=False))
            connection.commit()
        except SharedRateUnavailable:raise
        except (OSError,sqlite3.Error,ValueError,TypeError,OverflowError):raise SharedRateUnavailable() from None
        finally:
            if connection is not None:connection.close()
            if descriptor is not None:os.close(descriptor)

    def charge(self,client,kind,amount):
        if kind not in ('requests','outputs') or type(amount) is not int or not 1<=amount<=2**31-1:
            raise ValueError('invalid shared client rate charge')
        policy=(client.max_requests_per_minute,client.max_output_tokens_per_minute)
        if self.policies.get(client.name)!=policy:raise SharedRateUnavailable()
        index=0 if kind=='requests' else 1;limit=policy[index]
        if limit is None:return
        if amount>limit:raise ValueError('declared charge exceeds whole client rate allowance')
        wall=self.clock()
        if type(wall) not in (int,float) or not math.isfinite(wall) or not 0<=wall<=2**40:raise SharedRateUnavailable()
        retry=None
        with self._transaction() as connection:
            row=connection.execute('SELECT request_limit,output_limit,last_seen FROM clients WHERE name=?',(client.name,)).fetchone()
            if (row is None or row[:2]!=policy or type(row[2]) not in (int,float)
                    or not math.isfinite(row[2]) or not 0<=row[2]<=2**40):raise SharedRateUnavailable()
            now=max(wall,row[2])
            connection.execute('UPDATE clients SET last_seen=? WHERE name=?',(now,client.name))
            connection.execute('DELETE FROM buckets WHERE client=? AND expires<=?',(client.name,now))
            history=connection.execute('SELECT expires,requests,outputs FROM buckets WHERE client=? ORDER BY expires LIMIT 62',(client.name,)).fetchall()
            if len(history)>61 or any(type(expiry) is not int or not now<expiry<=math.floor(now)+61
                    or any(type(count) is not int or count<0 or count>(bound or 0) for count,bound in zip(counts,policy))
                    for expiry,*counts in history):raise SharedRateUnavailable()
            used=sum(record[index+1] for record in history)
            if used>limit:raise SharedRateUnavailable()
            if amount>limit-used:
                for expiry,*counts in history:
                    used-=counts[index]
                    if amount<=limit-used:
                        retry=max(1,math.ceil(expiry-wall));break
            else:
                expires=math.floor(now)+61
                requests,outputs=(amount,0) if index==0 else (0,amount)
                connection.execute('INSERT INTO buckets VALUES(?,?,?,?) ON CONFLICT(client,expires) DO UPDATE SET requests=requests+excluded.requests, outputs=outputs+excluded.outputs',
                    (client.name,expires,requests,outputs))
        if retry is not None:raise SharedRateRejected(retry)
