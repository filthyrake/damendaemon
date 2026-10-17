"""Explicit queued native GLM host HTTP admission."""
from .glm_host_serving import HostGlmReferenceRuntime
from .queued_native_host_serving import QueuedNativeHostReferenceRuntime


class QueuedHostGlmReferenceRuntime(QueuedNativeHostReferenceRuntime,HostGlmReferenceRuntime):
    pass
