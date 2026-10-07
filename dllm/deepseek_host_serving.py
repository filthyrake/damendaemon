"""Explicit native DeepSeek text/thinking/DSML host HTTP admission."""
from .deepseek_host_generation import HostDeepSeekGenerator
from .native_host_serving import NativeHostReferenceRuntime


class HostDeepSeekReferenceRuntime(NativeHostReferenceRuntime):
    runner_type=HostDeepSeekGenerator
    model_type='deepseek_v4'
