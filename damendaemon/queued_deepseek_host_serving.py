"""Explicit bounded serialized native DeepSeek host HTTP admission."""
from .deepseek_host_serving import HostDeepSeekReferenceRuntime
from .queued_native_host_serving import QueuedNativeHostReferenceRuntime


class QueuedHostDeepSeekReferenceRuntime(QueuedNativeHostReferenceRuntime,HostDeepSeekReferenceRuntime):
    pass
