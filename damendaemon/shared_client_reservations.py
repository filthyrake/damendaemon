"""Explicit same-host outstanding admission claims, protected by owner flocks.

Private local POSIX storage only. Locks have no expiry: an unlocked owner slot
is the only proof permitting crash reclamation. These are admission counts,
not GPU memory reservations. Never unlink or replace the store while in use.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
try:import fcntl
except ImportError:fcntl=None
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import threading

APP_ID=0x444C5253
MAX_OWNERS=128
MAX_CLAIMS=32768
MAX_BYTES=32*1024**2
SCHEMA=(
    'CREATE TABLE storage(kind TEXT PRIMARY KEY, device INTEGER NOT NULL, inode INTEGER NOT NULL)',
    'CREATE TABLE slots(slot INTEGER PRIMARY KEY CHECK(slot>=0 AND slot<128), device INTEGER NOT NULL, inode INTEGER NOT NULL)',
    'CREATE TABLE clients(name TEXT PRIMARY KEY, requests INTEGER NOT NULL CHECK(requests BETWEEN 1 AND 128), outputs INTEGER CHECK(outputs BETWEEN 1 AND 2147483647))',
    'CREATE TABLE owners(slot INTEGER PRIMARY KEY REFERENCES slots(slot), token TEXT NOT NULL UNIQUE)',
    'CREATE TABLE claims(id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES owners(token), client TEXT NOT NULL REFERENCES clients(name), requests INTEGER NOT NULL CHECK(requests IN (0,1)), outputs INTEGER NOT NULL CHECK(outputs BETWEEN 0 AND 2147483647), CHECK(requests+outputs>0))',
)

INDEXES=('CREATE INDEX claims_client ON claims(client)', 'CREATE INDEX claims_owner ON claims(owner)')


class SharedReservationUnavailable(RuntimeError):
    def __init__(self,store=None):
        super().__init__('shared client reservation accounting unavailable')
        # A failed mutation may have committed. Retain the owner and its claims
        # for explicit retry; never release the OS lock optimistically.
        self.store=store
        self.claim=None


class SharedReservationRejected(Exception):
    def __init__(self,kind):
        super().__init__('client outstanding '+kind+' limit reached')
        self.kind=kind


@dataclass(frozen=True,eq=False)
class Reservation:
    id: str
    client: str
    requests: int
    _owner: object=field(repr=False)
    _released: bool=field(default=False,repr=False)


class SharedClientReservations:
    """Bounded durable claims; callers must release every claim then close.

    Failed reserve operations retain their handle in ``claims``. Call release
    for these handles to resolve uncertainty, then retry close. Construction
    failures expose this owner as ``error.store`` when cleanup needs retry.
    No finalizer silently returns capacity. Forked copies cannot use an owner.
    """
    def __init__(self,path,clients):
        if fcntl is None:raise ValueError('shared reservations require POSIX owner locks')
        from .http_auth import ClientKey
        if (not isinstance(clients,(list,tuple)) or not 1<=len(clients)<=128
                or any(not isinstance(c,ClientKey) for c in clients)
                or len({c.name for c in clients})!=len(clients)):
            raise ValueError('shared reservations require bounded named clients')
        raw=Path(path).expanduser()
        if not raw.is_absolute():raise ValueError('shared reservation path must be absolute')
        self.path=raw.parent.resolve(strict=True)/raw.name
        self.directory=self.path.with_name(self.path.name+'.owners')
        self.policies={c.name:(c.max_inflight,c.max_reserved_output_tokens) for c in clients}
        self.pid=os.getpid();self.mutex=threading.RLock();self.claims={};self._claim_identity=object()
        self.token=secrets.token_hex(16);self.slot=None;self.descriptor=None
        self.closed=False;self.file_identity=None;self.directory_identity=None
        self.uncertain=set();self.closing=False
        try:
            self._parent()
            fd=os.open(self.path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
            try:self.file_identity=self._file(os.fstat(fd))
            finally:os.close(fd)
            with self._transaction(initialize=True) as connection:
                fresh=connection.execute('PRAGMA application_id').fetchone()[0]==0
                if fresh:
                    # Existing ownership evidence with a new database is unsafe.
                    os.mkdir(self.directory,0o700)
                    self.directory_identity=self._directory()
                    for sql in (*SCHEMA,*INDEXES):connection.execute(sql)
                    for slot in range(MAX_OWNERS):
                        fd=os.open(self.directory/str(slot),os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                        try:
                            info=os.fstat(fd);connection.execute('INSERT INTO slots VALUES(?,?,?)',(slot,info.st_dev,info.st_ino))
                            os.fsync(fd)
                        finally:os.close(fd)
                    connection.executemany('INSERT INTO storage VALUES(?,?,?)',
                        [('file',*self.file_identity),('directory',*self.directory_identity)])
                    connection.execute(f'PRAGMA application_id={APP_ID}')
                    connection.execute('PRAGMA user_version=1')
                    fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                else:
                    self.directory_identity=self._directory()
                    self._validate(connection)
                    if connection.execute('PRAGMA quick_check').fetchone()!=('ok',):raise SharedReservationUnavailable(self)
                with os.scandir(self.directory) as entries:
                    names=set()
                    for entry in entries:
                        names.add(entry.name)
                        if len(names)>MAX_OWNERS:raise SharedReservationUnavailable(self)
                if names!={str(i) for i in range(MAX_OWNERS)}:raise SharedReservationUnavailable(self)
                for slot in range(MAX_OWNERS):os.close(self._slot_file(connection,slot))
                for slot in range(MAX_OWNERS):
                    fd=self._slot_file(connection,slot)
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:os.close(fd);continue
                    except BaseException:os.close(fd);raise
                    self.slot,self.descriptor=slot,fd
                    old=connection.execute('SELECT token FROM owners WHERE slot=?',(slot,)).fetchone()
                    if old:
                        connection.execute('DELETE FROM claims WHERE owner=?',old)
                        connection.execute('DELETE FROM owners WHERE slot=?',(slot,))
                    connection.execute('INSERT INTO owners VALUES(?,?)',(slot,self.token))
                    break
                if self.slot is None:raise SharedReservationUnavailable(self)
                for name,policy in self.policies.items():
                    previous=connection.execute('SELECT requests,outputs FROM clients WHERE name=?',(name,)).fetchone()
                    if previous is None:
                        if connection.execute('SELECT count(*) FROM clients').fetchone()[0]>=128:raise SharedReservationUnavailable(self)
                        connection.execute('INSERT INTO clients VALUES(?,?,?)',(name,*policy))
                    elif previous!=policy:raise SharedReservationUnavailable(self)
        except Exception:
            # A constructor may have acquired an owner before a failed commit.
            # Keep it reachable, even if cleanup itself cannot be proven.
            if self.descriptor is not None:
                try:self.close()
                except Exception:raise SharedReservationUnavailable(self) from None
            raise SharedReservationUnavailable() from None

    def _parent(self):
        info=self.path.parent.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o022:
            raise SharedReservationUnavailable(self)

    def _file(self,info):
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077
                or info.st_nlink!=1 or info.st_size>MAX_BYTES):raise SharedReservationUnavailable(self)
        identity=info.st_dev,info.st_ino
        if self.file_identity is not None and identity!=self.file_identity:raise SharedReservationUnavailable(self)
        return identity

    def _directory(self):
        info=self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
            raise SharedReservationUnavailable(self)
        identity=info.st_dev,info.st_ino
        if self.directory_identity is not None and identity!=self.directory_identity:raise SharedReservationUnavailable(self)
        return identity

    def _slot_file(self,connection,slot):
        self._directory()
        expected=connection.execute('SELECT device,inode FROM slots WHERE slot=?',(slot,)).fetchone()
        fd=os.open(self.directory/str(slot),os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            info=os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077
                    or info.st_nlink!=1 or info.st_size!=0 or (info.st_dev,info.st_ino)!=expected):raise SharedReservationUnavailable(self)
            return fd
        except BaseException:os.close(fd);raise

    def _validate(self,connection):
        tables=connection.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        if {row[0] for row in tables}!=set(SCHEMA):raise SharedReservationUnavailable(self)
        indexes={row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")}
        if indexes!=set(INDEXES):raise SharedReservationUnavailable(self)
        if connection.execute("SELECT 1 FROM claims WHERE typeof(outputs)!='integer' OR typeof(requests)!='integer' OR requests NOT IN (0,1) OR outputs<0 OR outputs>2147483647 OR requests+outputs=0 OR length(id)!=32 LIMIT 1").fetchone():raise SharedReservationUnavailable(self)
        if connection.execute("SELECT 1 FROM owners WHERE length(token)!=32 LIMIT 1").fetchone():raise SharedReservationUnavailable(self)
        if connection.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('trigger','view')").fetchone()[0]:raise SharedReservationUnavailable(self)
        if dict((kind,(dev,ino)) for kind,dev,ino in connection.execute('SELECT * FROM storage'))!={'file':self.file_identity,'directory':self.directory_identity}:
            raise SharedReservationUnavailable(self)
        if (connection.execute('SELECT count(*) FROM slots').fetchone()[0]!=MAX_OWNERS
                or connection.execute('SELECT count(*) FROM clients').fetchone()[0]>128
                or connection.execute('SELECT count(*) FROM claims').fetchone()[0]>MAX_CLAIMS
                or connection.execute('PRAGMA foreign_key_check').fetchone() is not None):raise SharedReservationUnavailable(self)
        for name,policy in self.policies.items():
            row=connection.execute('SELECT requests,outputs FROM clients WHERE name=?',(name,)).fetchone()
            if row is not None and row!=policy:raise SharedReservationUnavailable(self)

    @contextmanager
    def _transaction(self,*,initialize=False):
        if os.getpid()!=self.pid or self.closed:raise SharedReservationUnavailable(self)
        connection=None;fd=None
        with self.mutex:
            try:
                self._parent()
                fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                self._file(os.fstat(fd))
                connection=sqlite3.connect(str(self.path),timeout=1,isolation_level=None)
                connection.execute('PRAGMA trusted_schema=OFF');connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA synchronous=FULL')
                if connection.execute('PRAGMA journal_mode').fetchone()[0]!='delete':raise SharedReservationUnavailable(self)
                if connection.execute('PRAGMA page_size').fetchone()[0]!=4096:raise SharedReservationUnavailable(self)
                connection.execute('PRAGMA max_page_count=8192');connection.execute('BEGIN IMMEDIATE')
                app=connection.execute('PRAGMA application_id').fetchone()[0];version=connection.execute('PRAGMA user_version').fetchone()[0]
                fresh=initialize and app==0 and version==0 and not connection.execute('SELECT count(*) FROM sqlite_master').fetchone()[0]
                if not fresh and (app!=APP_ID or version!=1):raise SharedReservationUnavailable(self)
                if not initialize:self._directory();self._validate(connection)
                yield connection
                self._file(os.stat(self.path,follow_symlinks=False));self._directory()
                self._commit(connection)
            except (SharedReservationRejected,SharedReservationUnavailable):raise
            except (OSError,sqlite3.Error,ValueError,TypeError,OverflowError):raise SharedReservationUnavailable(self) from None
            finally:
                if connection is not None:connection.close()
                if fd is not None:os.close(fd)

    def _commit(self,connection):connection.commit()

    @contextmanager
    def _owners(self,connection):
        held=[]
        try:
            own=connection.execute('SELECT token FROM owners WHERE slot=?',(self.slot,)).fetchone()
            if own!=(self.token,):raise SharedReservationUnavailable(self)
            for slot,token in connection.execute('SELECT slot,token FROM owners').fetchall():
                fd=self._slot_file(connection,slot)
                if token==self.token:os.close(fd);continue
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:os.close(fd);continue
                except BaseException:os.close(fd);raise
                held.append(fd)
                connection.execute('DELETE FROM claims WHERE owner=?',(token,))
                connection.execute('DELETE FROM owners WHERE slot=?',(slot,))
            yield
        finally:
            # Transaction serialization prevents another owner being registered
            # before this transaction commits, even after these locks close.
            for fd in held:os.close(fd)

    def _client(self,client):
        if self.policies.get(client.name)!=(client.max_inflight,client.max_reserved_output_tokens):raise SharedReservationUnavailable(self)
        return client.name

    def _capacity(self,connection,name,requests,outputs,exclude=None):
        used=connection.execute('SELECT coalesce(sum(requests),0),coalesce(sum(outputs),0) FROM claims WHERE client=? AND id!=?',(name,exclude or '')).fetchone()
        cap=self.policies[name]
        if requests>cap[0]-used[0]:raise SharedReservationRejected('requests')
        if cap[1] is not None and outputs>cap[1]-used[1]:raise SharedReservationRejected('outputs')

    def reserve(self,client,*,outputs=0,requests=1):
        name=self._client(client)
        if type(requests) is not int or requests not in (0,1) or type(outputs) is not int or not 0<=outputs<=2**31-1 or requests+outputs==0:
            raise ValueError('invalid outstanding reservation')
        with self.mutex:
            if self.closed or self.closing or self.uncertain or os.getpid()!=self.pid:raise SharedReservationUnavailable(self)
            claim=Reservation(secrets.token_hex(16),name,requests,self._claim_identity);self.claims[claim.id]=claim
            try:
                with self._transaction() as connection,self._owners(connection):
                    self._capacity(connection,name,requests,outputs)
                    if connection.execute('SELECT count(*) FROM claims').fetchone()[0]>=MAX_CLAIMS:raise SharedReservationUnavailable(self)
                    connection.execute('INSERT INTO claims VALUES(?,?,?,?,?)',(claim.id,self.token,name,requests,outputs))
            except SharedReservationRejected:
                self.claims.pop(claim.id);raise
            except BaseException as error:
                self.uncertain.add(claim.id)
                if isinstance(error,SharedReservationUnavailable):error.claim=claim
                raise
            return claim

    def _claim(self,claim):
        if not isinstance(claim,Reservation) or self.claims.get(claim.id) is not claim:raise ValueError('reservation does not belong to this owner')

    def resize(self,claim,outputs):
        if type(outputs) is not int or not 0<=outputs<=2**31-1:raise ValueError('invalid outstanding output reservation')
        with self.mutex:
            self._claim(claim)
            if claim.requests+outputs==0:raise ValueError('empty reservation must be released')
            self.uncertain.add(claim.id)
            try:
                with self._transaction() as connection,self._owners(connection):
                    row=connection.execute('SELECT owner,client,requests FROM claims WHERE id=?',(claim.id,)).fetchone()
                    if row!=(self.token,claim.client,claim.requests):raise SharedReservationUnavailable(self)
                    self._capacity(connection,claim.client,claim.requests,outputs,claim.id)
                    connection.execute('UPDATE claims SET outputs=? WHERE id=?',(outputs,claim.id))
            except SharedReservationRejected:
                self.uncertain.discard(claim.id);raise
            self.uncertain.discard(claim.id)

    def release(self,claim):
        with self.mutex:
            # Only this exact owner-issued object can perform idempotent release.
            if isinstance(claim,Reservation) and claim._owner is self._claim_identity and claim._released:return
            self._claim(claim)
            self.uncertain.add(claim.id)
            with self._transaction() as connection,self._owners(connection):
                row=connection.execute('SELECT owner FROM claims WHERE id=?',(claim.id,)).fetchone()
                if row is not None and row!=(self.token,):raise SharedReservationUnavailable(self)
                connection.execute('DELETE FROM claims WHERE id=?',(claim.id,))
            self.uncertain.discard(claim.id)
            self.claims.pop(claim.id)
            object.__setattr__(claim,'_released',True)

    def close(self):
        with self.mutex:
            if self.closed:return
            if os.getpid()!=self.pid:raise SharedReservationUnavailable(self)
            if self.claims:raise RuntimeError('release outstanding reservations before closing owner')
            self.closing=True
            if self.descriptor is not None:
                with self._transaction() as connection:
                    os.close(self._slot_file(connection,self.slot))
                    row=connection.execute('SELECT token FROM owners WHERE slot=?',(self.slot,)).fetchone()
                    if row is not None and row!=(self.token,):raise SharedReservationUnavailable(self)
                    if connection.execute('SELECT count(*) FROM claims WHERE owner=?',(self.token,)).fetchone()[0]:raise SharedReservationUnavailable(self)
                    connection.execute('DELETE FROM owners WHERE token=?',(self.token,))
                os.close(self.descriptor);self.descriptor=None
            self.closed=True
