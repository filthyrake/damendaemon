"""Stage-local DeepSeek packed checkpoint construction for token-bearing hosts."""
import copy
from .cpu_host_pipeline_build import CpuHostPipelineBuildError
from .deepseek_reference_loader import load_reference_stage
from .deepseek_stage_slots import DeepSeekStageSlots


class DeepSeekHostPipelineBuildError(CpuHostPipelineBuildError):
    """Retains claimed links and failed stage release for explicit retry."""


class _OwnedDeepSeekHostStage(DeepSeekStageSlots):
    def __init__(self,stage,context,slots,**options):
        super().__init__(stage,context,slots,**options)
        self._config=copy.deepcopy(stage.config);self._range=(stage.start,stage.end)

    @property
    def c(self):return copy.deepcopy(self._config)

    @property
    def layer_start(self):return self._range[0]

    @property
    def layer_end(self):return self._range[1]

    def close(self):
        with self.lock:
            if self.stage is None:return
            super().close();self.stage=None


def _load_stage(path,start,end,*,max_weight_bytes,projection_workspace_bytes,max_workspace_bytes,max_output_bytes):
    return load_reference_stage(path,start,end,max_weight_bytes=max_weight_bytes,
        workspace_bytes=projection_workspace_bytes,max_output_bytes=max_output_bytes)


def build_deepseek_host_pipeline_worker(path,topology,index,transports,*,epoch,verify_source=False,
        identity_max_bytes=2*1024**3,identity_timeout=60,max_weight_bytes=256*1024**2,
        workspace_bytes=8*1024**2,max_output_bytes=64*1024**2,max_cache_bytes=256*1024**2):
    from .reference_host_pipeline_build import build_reference_host_worker
    return build_reference_host_worker(path,topology,index,transports,model_type='deepseek_v4',
        quantization='fp4_fp8_reference',load_stage=_load_stage,owned_stage=_OwnedDeepSeekHostStage,
        error_type=DeepSeekHostPipelineBuildError,epoch=epoch,verify_source=verify_source,
        identity_max_bytes=identity_max_bytes,identity_timeout=identity_timeout,max_weight_bytes=max_weight_bytes,
        projection_workspace_bytes=workspace_bytes,max_output_bytes=max_output_bytes,max_cache_bytes=max_cache_bytes)
