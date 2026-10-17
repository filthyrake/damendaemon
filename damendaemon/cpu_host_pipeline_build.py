"""Stage-local snapshot construction for explicitly planned CPU host workers."""
from contextlib import ExitStack
from .cpu_hybrid_checkpoint import load_cpu_stage_checkpoint
from .cpu_hybrid_stage import CpuHybridStage
from .cpu_snapshot_identity import fingerprint
from .host_pipeline_worker import HostPipelineCoordinator,HostPipelineWorker
from .host_tensor_channel import HostTensorChannel
from .host_topology import HostTopology
from .planned_host_stage_channel import PlannedHostStageChannel
from .stage_protocol import StageState


class CpuHostPipelineBuildError(RuntimeError):
    """Own every claimed resource until cleanup succeeds; close is retryable."""
    def __init__(self,message,transports,channels,stage):
        super().__init__(message)
        self.transports,self.channels,self.stage=tuple(transports),tuple(channels),stage
        self.closed=False

    def close(self):
        if self.closed:return
        errors=[]
        for owner in (*self.channels,*self.transports):
            try:owner.close()
            except BaseException as error:errors.append(error)
        if self.stage is not None:
            try:(getattr(self.stage,'release_partial',None) or self.stage.close)()
            except BaseException as error:errors.append(error)
        if errors:
            if len(errors)>1:errors[0].add_note('additional CPU topology cleanup failures; retain build error and retry close')
            raise errors[0]
        self.closed=True


def _claim_transports(topology,index,transports,epoch):
    """Atomically claim validated fresh ordered links for one stage builder."""
    StageState(epoch,topology.context,topology.slots,context_limit=topology.context_limit)
    remote=list(range(1,len(topology.ranges))) if index==0 else [index]
    rank=0 if index==0 else 1
    if (not isinstance(transports,(list,tuple)) or len(transports)!=len(remote)
            or len({id(t) for t in transports})!=len(transports)
            or any(not isinstance(t,HostTensorChannel) or t.rank!=rank
                   or t.identity!=topology.stage_identity(i) or t.max_frame_bytes!=topology.frame_bytes
                   for t,i in zip(transports,remote))):
        raise ValueError('CPU topology build requires matching ordered fresh transports')
    transports=tuple(transports)
    with ExitStack() as locks:
        for transport in sorted(transports,key=id):locks.enter_context(transport.lock)
        if any(t.ready or t.closed or t.failed or getattr(t,'_cpu_worker_builder_owner',None) is not None for t in transports):
            raise ValueError('CPU topology build transport is not fresh or already claimed')
        owner=object()
        for transport in transports:transport._cpu_worker_builder_owner=owner
    return transports


def build_cpu_host_pipeline_worker(path,topology,index,transports,*,epoch,verify_source=False,
        identity_max_bytes=2*1024**3,identity_timeout=60,max_weight_bytes=256*1024**2,
        loader_workspace_bytes=64*1024**2,**stage_options):
    """Claim fresh links and load only one assigned CPU layer range.

    Index zero needs links ordered by remote index 1..N-1; other indices need
    their single rank-one link. Authenticate sockets before constructing these
    transports. No listener, thread, handshake or device selection occurs here.
    Invalid controls leave links caller-owned. After claim, failures clean all
    links and partial storage; retain an unclosed build error for cleanup retry.
    Optional full-snapshot fingerprinting verifies topology.source_identity;
    otherwise that value is an operator assertion, not byte verification.
    """
    if (not isinstance(topology,HostTopology) or topology.config.get('model_type')!='qwen3_5_text' or type(index) is not int
            or not 0<=index<len(topology.ranges) or type(verify_source) is not bool
            or type(identity_max_bytes) is not int or not 1<=identity_max_bytes<=2**40
            or type(identity_timeout) not in (int,float) or not 0<identity_timeout<=3600
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_weight_bytes,loader_workspace_bytes))):
        raise ValueError('invalid CPU topology build controls')
    transports=_claim_transports(topology,index,transports,epoch)
    remote=list(range(1,len(topology.ranges))) if index==0 else [index]
    channels=[];stage=None;model=None
    try:
        verified=None
        if verify_source:
            verified=fingerprint(path,max_bytes=identity_max_bytes,timeout=identity_timeout)
            if bytes.fromhex(verified.model_id)!=topology.source_identity:
                raise ValueError('CPU topology source identity differs from snapshot bytes')
            path=verified.root
        start,end=topology.ranges[index]
        model=load_cpu_stage_checkpoint(path,start,end,quantization=topology.quantization,max_context=topology.context,
            max_weight_bytes=max_weight_bytes,max_workspace_bytes=loader_workspace_bytes)
        if model.config!=topology.config:raise ValueError('CPU snapshot configuration differs from topology')
        if verified is not None:verified.verify_unchanged()
        for transport,remote_index in zip(transports,remote):
            state=StageState(epoch,topology.context,topology.slots,context_limit=topology.context_limit)
            channels.append(PlannedHostStageChannel(transport,state,topology,remote_index))
        stage=CpuHybridStage.__new__(CpuHybridStage)
        stage.__init__(model,start,end,topology.slots,max_weight_bytes=max_weight_bytes,**stage_options)
        if verified is not None:verified.verify_unchanged()
        worker=(HostPipelineCoordinator(stage,topology,channels) if index==0 else HostPipelineWorker(stage,channels[0]))
        worker.model_identity_verified=verify_source
        worker.source_model_id=topology.source_identity.hex()
        return worker
    except BaseException as error:
        failure=CpuHostPipelineBuildError(f'CPU topology construction failed: {type(error).__name__}: {error}',transports,channels,stage)
    model=None
    try:failure.close()
    except BaseException as error:
        failure.add_note(f'cleanup incomplete; retain this error and retry close: {type(error).__name__}: {error}')
    raise failure
