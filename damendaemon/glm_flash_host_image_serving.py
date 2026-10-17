"""Explicit native Flash image/tool HTTP owners for prepared CPU host stages."""
from .glm_flash_host_image_generation import HostFlashImageGenerator
from .native_host_serving import NativeHostReferenceRuntime
from .queued_native_host_serving import QueuedNativeHostReferenceRuntime


class HostFlashImageReferenceRuntime(NativeHostReferenceRuntime):
    runner_type=HostFlashImageGenerator
    model_type='glm5_next'
    supports_images=True


class QueuedHostFlashImageReferenceRuntime(QueuedNativeHostReferenceRuntime,HostFlashImageReferenceRuntime):
    pass
