"""Explicit Flash native host HTTP admission; text and tools only."""
from .glm_flash_host_generation import HostFlashGenerator
from .native_host_serving import NativeHostReferenceRuntime


class HostFlashReferenceRuntime(NativeHostReferenceRuntime):
    runner_type=HostFlashGenerator
    model_type='glm5_next'
