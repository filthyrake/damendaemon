"""Explicit GLM native host HTTP admission."""
from .glm_host_generation import HostGlmGenerator
from .native_host_serving import NativeHostReferenceRuntime


class HostGlmReferenceRuntime(NativeHostReferenceRuntime):
    runner_type=HostGlmGenerator
    model_type='glm_moe_dsa'
