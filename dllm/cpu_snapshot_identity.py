"""Bounded byte identity for explicit local CPU snapshot verification."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .download import INDEX,snapshot_files
from .safetensor_header import _unique_fields

FORMAT='dllm-cpu-snapshot-v1'
_REQUIRED=('config.json','tokenizer.json')
_OPTIONAL=('generation_config.json','tokenizer_config.json',INDEX)
_METADATA_LIMITS={'config.json':2**20,'tokenizer.json':32*2**20,
                  'generation_config.json':2**20,'tokenizer_config.json':2**20,INDEX:16*2**20}


def _stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


@contextmanager
def _root(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:yield fd
    finally:os.close(fd)


@contextmanager
def _reader(root,name):
    parts=name.split('/')
    if (not 1<=len(parts)<=16 or len(name.encode())>1024 or any(p in ('','.','..') for p in parts)
            or any(c in name for c in '\\*?[]\0')):raise ValueError('invalid snapshot identity path')
    directories=[];parent=root;fd=None
    try:
        for part in parts[:-1]:
            parent=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);directories.append(parent)
        fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('snapshot identity requires regular files')
        with os.fdopen(fd,'rb') as reader:
            fd=None;yield reader
    finally:
        if fd is not None:os.close(fd)
        for directory in reversed(directories):os.close(directory)


def _inventory(root):
    names=[]
    with os.scandir(root) as entries:
        for entry in entries:
            if len(names)>=1024:raise ValueError('snapshot directory exceeds entry limit')
            names.append(entry.name)
    if any(name not in names for name in _REQUIRED):raise ValueError('CPU identity needs config.json and tokenizer.json')
    index=None
    if INDEX in names:
        with _reader(root,INDEX) as reader:raw=reader.read(_METADATA_LIMITS[INDEX]+1)
        if len(raw)>_METADATA_LIMITS[INDEX]:raise ValueError('snapshot identity index exceeds byte limit')
        index=json.loads(raw,object_pairs_hook=_unique_fields)
        if isinstance(index,dict) and isinstance(index.get('weight_map'),dict):
            names.extend(name for name in index['weight_map'].values() if isinstance(name,str))
    shards=[name for name in snapshot_files(names,index) if name.endswith('.safetensors')]
    if not 1<=len(shards)<=256:raise ValueError('snapshot identity requires 1..256 shards')
    return tuple(sorted(set(shards)|set(_REQUIRED)|{name for name in _OPTIONAL if name in names}))


@dataclass(frozen=True)
class SnapshotIdentity:
    root: Path
    root_identity: tuple
    files: tuple
    stamps: tuple
    model_id: str
    total_bytes: int

    def report(self):
        return dict(format=FORMAT,model_id=self.model_id,total_bytes=self.total_bytes,
                    files=[dict(path=name,bytes=size,sha256=digest) for name,size,digest in self.files])

    def verify_unchanged(self):
        """Fail if inventory/inode/size/mtime/ctime changed; no payload re-read."""
        with _root(self.root) as root:
            info=os.fstat(root)
            if (info.st_dev,info.st_ino)!=self.root_identity or _inventory(root)!=tuple(x[0] for x in self.files):
                raise ValueError('verified CPU snapshot inventory changed')
            for (name,_,_),expected in zip(self.files,self.stamps):
                with _reader(root,name) as reader:
                    if _stamp(os.fstat(reader.fileno()))!=expected:raise ValueError('verified CPU snapshot file changed')


def fingerprint(path,*,max_bytes=2*1024**3,timeout=60):
    """Hash each selected byte once with 1 MiB read buffers and a total deadline.

    File layout and metadata bytes are part of identity, not just tensor values.
    No tensor decoding, model loading, network access or credential reads occur.
    Symlinks below the resolved snapshot root are rejected, including cache links.
    """
    if type(max_bytes) is not int or not 1<=max_bytes<=2**40 or type(timeout) not in (int,float) or not 0<timeout<=3600:
        raise ValueError('invalid snapshot identity byte/time bounds')
    deadline=time.monotonic()+timeout
    def check_time():
        if time.monotonic()>deadline:raise ValueError('snapshot identity deadline exceeded')
    path=Path(path).resolve(strict=True);records=[];stamps=[];total=0
    with _root(path) as root:
        root_stat=os.fstat(root);root_identity=(root_stat.st_dev,root_stat.st_ino)
        names=_inventory(root)
        # Reject aggregate and individual budgets before streaming shard payloads.
        for name in names:
            check_time()
            with _reader(root,name) as reader:info=os.fstat(reader.fileno())
            if info.st_size<1 or info.st_size>_METADATA_LIMITS.get(name,max_bytes):raise ValueError('snapshot identity file exceeds byte limit')
            total+=info.st_size
            if total>max_bytes:raise ValueError('snapshot identity exceeds aggregate byte limit')
            stamps.append(_stamp(info))
        for name,expected in zip(names,stamps):
            check_time();digest=hashlib.sha256();remaining=expected[2]
            with _reader(root,name) as reader:
                if _stamp(os.fstat(reader.fileno()))!=expected:raise ValueError('snapshot changed before identity read')
                while remaining:
                    check_time();chunk=reader.read(min(1024**2,remaining))
                    if not chunk:raise ValueError('snapshot truncated during identity read')
                    digest.update(chunk);remaining-=len(chunk)
                if reader.read(1) or _stamp(os.fstat(reader.fileno()))!=expected:raise ValueError('snapshot changed during identity read')
            records.append((name,expected[2],digest.hexdigest()))
    manifest=dict(format=FORMAT,files=[dict(path=n,bytes=s,sha256=d) for n,s,d in records])
    model_id=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    result=SnapshotIdentity(path,root_identity,tuple(records),tuple(stamps),model_id,total)
    result.verify_unchanged();check_time()
    return result
