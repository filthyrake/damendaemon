"""Explicit queued native Flash host HTTP admission."""
from .glm_flash_host_serving import HostFlashReferenceRuntime
from .queued_native_host_serving import QueuedNativeHostReferenceRuntime


class QueuedHostFlashReferenceRuntime(QueuedNativeHostReferenceRuntime,HostFlashReferenceRuntime):
    pass
