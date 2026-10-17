"""Explicit stage-local Flash snapshot construction for prepared host transports."""
import copy
from .cpu_host_pipeline_build import CpuHostPipelineBuildError
from .glm_flash_reference_loader import load_reference_stage
from .glm_flash_stage_slots import FlashStageSlots


class FlashHostPipelineBuildError(CpuHostPipelineBuildError):
    """All claimed links and unfinished stage releases remain owned for retry."""


class _OwnedFlashHostStage(FlashStageSlots):
    """The builder owns these checkpoint weights; release them after cache close."""
    def __init__(self, stage, context, slots, **options):
        super().__init__(stage, context, slots, **options)
        self._config = copy.deepcopy(stage.config)
        self._range = (stage.start, stage.end)

    @property
    def c(self): return copy.deepcopy(self._config)

    @property
    def layer_start(self): return self._range[0]

    @property
    def layer_end(self): return self._range[1]

    def close(self):
        with self.lock:
            if self.stage is None: return
            super().close()
            self.stage = None


def build_glm_flash_host_pipeline_worker(path,topology,index,transports,**options):
    from .reference_host_pipeline_build import build_reference_host_worker
    return build_reference_host_worker(path,topology,index,transports,model_type='glm5_next',
        load_stage=load_reference_stage,owned_stage=_OwnedFlashHostStage,error_type=FlashHostPipelineBuildError,**options)
