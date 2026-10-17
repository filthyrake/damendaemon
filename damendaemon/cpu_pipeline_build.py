"""Owned CPU reference construction, separate from accelerator/serving admission."""
from .cpu_hybrid_stage import CpuHybridStage
from .cpu_stage_plan import plan_cpu_stages
from .hybrid_pipeline import HybridPipeline


class CpuPipelineBuildError(RuntimeError):
    """Retain partial stages until cleanup succeeds; callers can retry close."""
    def __init__(self,message,stages):
        super().__init__(message)
        self.stages=tuple(stages)
        self.closed=False

    def close(self):
        if self.closed:return
        for stage in self.stages:stage.release_partial()
        self.closed=True


def build_cpu_pipeline(model,budgets,*,max_batch=4,**stage_options):
    """Construct a prepared reference pipeline under explicit storage budgets.

    No model loading, device probing, serving registration or live memory claims.
    Reserve source storage and unrelated process memory before passing budgets.
    On failure retain CpuPipelineBuildError while closed is false and retry close.
    """
    placement=plan_cpu_stages(model,budgets,max_batch=max_batch,**stage_options)
    if not placement['storage_plan_found']:raise MemoryError(placement['reason'])
    stages=[]
    try:
        for item in placement['stages']:
            stage=CpuHybridStage.__new__(CpuHybridStage)
            stages.append(stage)  # Own even a constructor that fails halfway.
            stage.__init__(model,item['layer_start'],item['layer_end'],max_batch,**stage_options)
            if stage.required_memory_bytes>item['cpu_budget_bytes']:
                raise MemoryError('CPU stage admission changed during construction')
        pipeline=HybridPipeline(stages)
        pipeline.model=model
        pipeline.placement=placement
        pipeline.devices=tuple('cpu' for _ in stages)
        pipeline.kernel_backend='cpu-reference-hybrid-pipeline-'+model.quantization
        return pipeline
    except BaseException as error:
        failure=CpuPipelineBuildError(f'CPU pipeline construction failed: {type(error).__name__}: {error}',stages)
    # Leave the original traceback (and temporary tensors) out of retained errors.
    try:failure.close()
    except BaseException as error:
        failure.add_note(f'cleanup incomplete; retain this error and retry close: {type(error).__name__}: {error}')
    raise failure
