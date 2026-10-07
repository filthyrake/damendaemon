"""Explicit stage-local GLM snapshot construction for prepared host transports."""
import copy
from .cpu_host_pipeline_build import _claim_transports, CpuHostPipelineBuildError
from .cpu_snapshot_identity import fingerprint
from .glm_reference_loader import load_reference_stage
from .glm_stage_slots import GlmStageSlots
from .host_topology import HostTopology
from .host_pipeline_worker import HostPipelineCoordinator, HostPipelineWorker
from .planned_host_stage_channel import PlannedHostStageChannel
from .stage_protocol import StageState


class GlmHostPipelineBuildError(CpuHostPipelineBuildError):
    """All claimed links and unfinished stage releases remain owned for retry."""


class _OwnedGlmHostStage(GlmStageSlots):
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


def build_glm_host_pipeline_worker(path, topology, index, transports, *, epoch, verify_source=False,
        identity_max_bytes=2*1024**3, identity_timeout=60, max_weight_bytes=256*1024**2,
        projection_workspace_bytes=8*1024**2, max_workspace_bytes=64*1024**2,
        max_output_bytes=64*1024**2, max_cache_bytes=256*1024**2):
    """Claim authenticated links, then load only this process's assigned range.

    Optional byte verification hashes the complete bounded snapshot, including
    tokenizer metadata, and rechecks its identity after loading/construction.
    No socket creation, handshake, listener or accelerator selection occurs here.
    Controls rejected before claim leave transports with their original owner.
    """
    bounds=(max_weight_bytes,projection_workspace_bytes,max_workspace_bytes,max_output_bytes,max_cache_bytes)
    if (not isinstance(topology,HostTopology) or topology.config.get('model_type')!='glm_moe_dsa'
            or topology.quantization!='fp8_reference' or type(index) is not int
            or not 0<=index<len(topology.ranges) or type(verify_source) is not bool
            or type(identity_max_bytes) is not int or not 1<=identity_max_bytes<=2**40
            or type(identity_timeout) not in (int,float) or not 0<identity_timeout<=3600
            or any(type(value) is not int or not 1<=value<=2**31 for value in bounds)):
        raise ValueError('invalid GLM host pipeline build controls')
    transports=_claim_transports(topology,index,transports,epoch)
    channels=[];stage=None;model=None
    try:
        verified=None
        if verify_source:
            verified=fingerprint(path,max_bytes=identity_max_bytes,timeout=identity_timeout)
            if bytes.fromhex(verified.model_id)!=topology.source_identity:
                raise ValueError('GLM topology source identity differs from snapshot bytes')
            path=verified.root
        start,end=topology.ranges[index]
        model=load_reference_stage(path,start,end,max_weight_bytes=max_weight_bytes,
            projection_workspace_bytes=projection_workspace_bytes,max_workspace_bytes=max_workspace_bytes,
            max_output_bytes=max_output_bytes)
        if model.config!=topology.config: raise ValueError('GLM snapshot configuration differs from topology')
        if verified is not None: verified.verify_unchanged()
        remote=list(range(1,len(topology.ranges))) if index==0 else [index]
        for transport,remote_index in zip(transports,remote):
            state=StageState(epoch,topology.context,topology.slots,context_limit=topology.context_limit)
            channels.append(PlannedHostStageChannel(transport,state,topology,remote_index))
        stage=_OwnedGlmHostStage(model,topology.context,topology.slots,max_cache_bytes=max_cache_bytes)
        if verified is not None: verified.verify_unchanged()
        worker=(HostPipelineCoordinator(stage,topology,channels) if index==0 else HostPipelineWorker(stage,channels[0]))
        worker.model_identity_verified=verify_source
        worker.source_model_id=topology.source_identity.hex()
        return worker
    except BaseException as error:
        failure=GlmHostPipelineBuildError(f'GLM host construction failed: {type(error).__name__}: {error}',transports,channels,stage)
    model=None
    try: failure.close()
    except BaseException as error:
        failure.add_note(f'cleanup incomplete; retain build error and retry close: {type(error).__name__}: {error}')
    raise failure
