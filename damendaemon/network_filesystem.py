"""Filesystem-class detection for checkpoint identity on network mounts.

Local filesystems keep the repository's strict file-identity contracts
(size/mtime/ctime metadata included). Some network filesystems update ctime
metadata when files are merely read, which made unchanged checkpoints look
modified and aborted shard discovery mid-scan. On such filesystems identity
drops ctime and keeps device/inode/size/mtime, which still detects the
ordinary replacement and modification the checkpoint receipts document as
their threat model, while tolerating read-side metadata jitter. The tolerance
is per filesystem class, never global.

Classification is host-local: the mount table is parsed once per process and
each network mountpoint is stated once to map its st_dev. Identity calls then
classify by device number, so callers that only hold a file descriptor or a
stat result need no extra path plumbing. Unmapped devices fail closed to the
strict local contract: an unclassifiable filesystem behaves exactly as before
this module existed. Network shares mounted after process start are likewise
unclassified until the process restarts — a fail-closed limitation, never a
silent weakening.
"""
import os
import sys
from functools import lru_cache


NETWORK_FILESYSTEMS = frozenset({
    'nfs', 'nfs4', 'cifs', 'smbfs', 'smb', 'smb2', 'afpfs', 'sshfs',
    'fuse.sshfs', 'webdav', 'davfs', 'ncpfs',
})


def _decode_mount_name(name):
    # /proc/mounts escapes whitespace and control characters octally.
    return (name.replace('\\040', ' ').replace('\\011', '\t')
                .replace('\\012', '\n').replace('\\015', '\r').replace('\\134', '\\'))


def _linux_mount_table():
    entries = []
    try:
        with open('/proc/mounts', 'r', encoding='utf-8', errors='replace') as table:
            for line in table:
                fields = line.split()
                if len(fields) >= 3:
                    entries.append((_decode_mount_name(fields[1]), fields[2]))
    except OSError:
        pass
    return entries


def _macos_mount_table():
    import subprocess
    entries = []
    try:
        completed = subprocess.run(['/sbin/mount'], capture_output=True,
                                    timeout=10, text=True)
    except (OSError, subprocess.SubprocessError):
        return entries
    for line in completed.stdout.splitlines():
        head, separator, options = line.partition(' (')
        if not separator or not options:
            continue
        device, separator, mountpoint = head.rpartition(' on ')
        if not separator or not mountpoint:
            continue
        entries.append((mountpoint, options.split(',')[0].strip()))
    return entries


def _mount_table():
    if os.path.exists('/proc/mounts'):
        return _linux_mount_table()
    if sys.platform == 'darwin':
        return _macos_mount_table()
    return []


@lru_cache(maxsize=1)
def _network_devices():
    """Device numbers of mounted network filesystems; empty when none resolve."""
    devices = set()
    for mountpoint, kind in _mount_table():
        if kind.lower() not in NETWORK_FILESYSTEMS:
            continue
        try:
            devices.add(os.stat(mountpoint).st_dev)
        except OSError:
            continue
    return frozenset(devices)


def network_device(st_dev):
    """True when a device number belongs to a mounted network filesystem.

    Unmapped devices are strictly local: a filesystem this process could not
    classify keeps the full size/mtime/ctime identity contract.
    """
    return st_dev in _network_devices()


def is_network_filesystem(path):
    """Classify a path's filesystem by device; unreadable paths are local."""
    try:
        return network_device(os.stat(path).st_dev)
    except OSError:
        return False
