"""Explicit stage-local CUDA construction for already authenticated host links."""
from copy import deepcopy
import torch
from .host_topology import HostTopology
from .cpu_host_pipeline_build import CpuHostPipelineBuildError
from .reference_host_pipeline_build import build_reference_host_worker
from .cuda_stage_placement import place_cuda_stage
from .cuda_stage_slots import CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots
from .cuda_host_stage import CudaHostStageSlots
from .glm_reference_loader import load_reference_stage as load_glm
from .glm_flash_reference_loader import load_reference_stage as load_flash
from .deepseek_host_pipeline_build import _load_stage as load_deepseek


class CudaHostPipelineBuildError(CpuHostPipelineBuildError):
    """Retain claimed links and any unfinished stage cleanup for retry."""


class _OwnedCudaHostStage(CudaHostStageSlots):
    def __init__(self,slots,*,max_transfer_bytes):
        config=deepcopy(slots.c);layer_range=(slots.layer_start,slots.layer_end);device=slots.actual_device
        super().__init__(slots,max_transfer_bytes=max_transfer_bytes)
        self._config=config;self._range=layer_range;self._device=device

    @property
    def c(self):return deepcopy(self._config)
    @property
    def layer_start(self):return self._range[0]
    @property
    def layer_end(self):return self._range[1]
    @property
    def actual_device(self):return self._device

    def close(self):
        with self.lock:
            if self.slots.stage is None:return
            super().close()
            self.slots.stage=None


def build_cuda_host_pipeline_worker(path,topology,index,transports,*,device,max_device_bytes,max_transfer_bytes,
        device_workspace_bytes=256*1024**2,**options):
    """Load only the assigned CPU range, then explicitly place/own it on CUDA.

    CPU loader/identity/cache controls are forwarded unchanged. Invalid device
    controls precede link claim; after claim, normal builder cleanup owns links.
    No socket, process, handshake, automatic device choice or peer access occurs.
    """
    device=torch.device(device)
    if (not isinstance(topology,HostTopology) or device.type!='cuda' or device.index is None
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_device_bytes,max_transfer_bytes,device_workspace_bytes))):
        raise ValueError('invalid explicit CUDA host build controls')
    recipes={'glm_moe_dsa':(load_glm,CudaGlmStageSlots,'fp8_reference'),
             'glm5_next':(load_flash,CudaFlashStageSlots,'fp8_reference'),
             'deepseek_v4':(load_deepseek,CudaDeepSeekStageSlots,'fp4_fp8_reference')}
    family=topology.config.get('model_type')
    if family not in recipes:raise ValueError('unsupported CUDA host reference family')
    loader,kind,quantization=recipes[family]
    if device_workspace_bytes<2:raise ValueError('invalid CUDA outer workspace bound')
    def owned_stage(source,context,slots,**slot_options):
        placed=place_cuda_stage(source,device=device,max_bytes=max_device_bytes,workspace_bytes=device_workspace_bytes)
        owner=kind(placed.stage,context,slots,**slot_options)
        return _OwnedCudaHostStage(owner,max_transfer_bytes=max_transfer_bytes)
    return build_reference_host_worker(path,topology,index,transports,model_type=family,load_stage=loader,
        owned_stage=owned_stage,error_type=CudaHostPipelineBuildError,quantization=quantization,**options)
