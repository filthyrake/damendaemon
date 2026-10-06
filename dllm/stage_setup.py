"""Bounded startup status on torchrun's existing CPU rendezvous store.

This catches reported preparation failures before NCCL initialization. It does
not interrupt a blocked model loader/compiler or replace the process owner's
hard deadline. Use a new session name for each preparation within one agent run.
"""
from datetime import timedelta
import hashlib
import os
import time


class StageSetup:
    def __init__(self, store, rank, timeout=60):
        if type(rank) is not int or rank not in (0, 1) or not 0 < timeout <= 120:
            raise ValueError('invalid stage setup bounds')
        self.store, self.rank = store, rank
        self.deadline = time.monotonic() + timeout
        self.local_ready = False

    @classmethod
    def from_torchrun(cls, rank, session='initial', timeout=60):
        import torch.distributed as dist
        if (os.environ.get('TORCHELASTIC_USE_AGENT_STORE', '').lower() != 'true'
                or os.environ.get('WORLD_SIZE') != '2' or os.environ.get('RANK') != str(rank)):
            raise ValueError('stage setup requires a matching two-worker torchrun agent store')
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError('invalid stage setup session')
        identity = '\0'.join((os.environ.get('TORCHELASTIC_RUN_ID', ''),
                              os.environ.get('TORCHELASTIC_RESTART_COUNT', '0'), session))
        prefix = 'dllm-stage-setup-v1/' + hashlib.sha256(identity.encode()).hexdigest()
        store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                              world_size=None, is_master=False, timeout=timedelta(seconds=5))
        return cls(dist.PrefixStore(prefix, store), rank, timeout)

    def check_peer(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('stage preparation deadline exceeded')
        key = str(1 - self.rank)
        if not self.store.check([key]):return False
        status = self.store.get(key)
        if status == b'ready':return True
        if status.startswith(b'failed:') and len(status) <= 135:
            raise RuntimeError('peer stage preparation failed: ' + status[7:].decode('ascii', errors='replace'))
        raise ValueError('invalid peer stage preparation status')

    def ready(self):
        self.check_peer()
        self.store.set(str(self.rank), b'ready')
        self.local_ready = True

    def wait_ready(self):
        if not self.local_ready:raise RuntimeError('local stage preparation is not ready')
        while not self.check_peer():time.sleep(.05)

    def fail(self, error):
        # Publish only the exception type, not paths or arbitrary error content.
        label = type(error).__name__.encode('ascii', errors='replace')[:128]
        self.store.set(str(self.rank), b'failed:' + label)
