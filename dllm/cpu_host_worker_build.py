"""Owned stage-local CPU worker construction from a fresh host transport."""
from .cpu_hybrid_checkpoint import load_cpu_stage_checkpoint
from .cpu_hybrid_stage import CpuHybridStage
from .host_tensor_channel import HostTensorChannel
from .host_stage_channel import HostStageChannel
from .host_hybrid_worker import HostHybridWorker
from .stage_protocol import StageState,HEADER_WORDS


class CpuHostWorkerBuildError(RuntimeError):
    """Keep partial resources reachable until close succeeds."""
    def __init__(self,message,transport,channel,stage):
        super().__init__(message)
        self.transport,self.channel,self.stage=transport,channel,stage
        self.closed=False

    def close(self):
        if self.closed:return
        self.transport.close()
        if self.channel is not None:self.channel.close()
        if self.stage is not None:self.stage.release_partial()
        self.closed=True


def build_cpu_host_worker(path,start,end,transport,*,epoch,max_context=512,max_batch=4,context_limit=512,
                          quantization='none',max_weight_bytes=256*1024**2,loader_workspace_bytes=64*1024**2,
                          **stage_options):
    """Claim a fresh transport and construct its selected CPU stage and worker.

    Local control validation leaves invalid transport inputs caller-owned. After
    claiming, any failure closes transport and partial stage storage; retain a
    CpuHostWorkerBuildError with closed=False and retry close. Returned workers
    are fresh: the caller explicitly handshakes both peers. The transport retains
    its existing trust policy; use authenticated endpoint sockets on real links.
    No socket binding, connection or worker thread is created here.
    """
    state=StageState(epoch,max_context,max_batch,context_limit=context_limit)
    if (not isinstance(transport,HostTensorChannel) or type(start) is not int or type(end) is not int
            or not 0<=start<end<=256 or transport.max_frame_bytes<8*HEADER_WORDS
            or (transport.rank==0 and start!=0) or (transport.rank==1 and start==0)):
        raise ValueError('invalid CPU host-worker control or layer ownership')
    with transport.lock:
        if (transport.ready or transport.closed or transport.failed
                or getattr(transport,'_cpu_worker_builder_owner',None) is not None):
            raise ValueError('CPU host-worker transport must be fresh and unclaimed')
        transport._cpu_worker_builder_owner=object()
    channel=None;stage=None;model=None
    try:
        model=load_cpu_stage_checkpoint(path,start,end,quantization=quantization,max_context=max_context,
                                       max_weight_bytes=max_weight_bytes,max_workspace_bytes=loader_workspace_bytes)
        count=model.config['num_hidden_layers']
        if (transport.rank==0 and end>=count) or (transport.rank==1 and end!=count):
            raise ValueError('CPU host-worker range does not match its coordinator/output rank')
        channel=HostStageChannel(transport,state,vocab_size=model.config['vocab_size'])
        stage=CpuHybridStage.__new__(CpuHybridStage)
        stage.__init__(model,start,end,max_batch,max_weight_bytes=max_weight_bytes,**stage_options)
        return HostHybridWorker(channel,stage)
    except BaseException as error:
        failure=CpuHostWorkerBuildError(f'CPU host-worker construction failed: {type(error).__name__}: {error}',transport,channel,stage)
    model=None  # Do not retain loader locals/source tensors in a failure traceback.
    try:failure.close()
    except BaseException as error:
        failure.add_note(f'cleanup incomplete; retain this error and retry close: {type(error).__name__}: {error}')
    raise failure
