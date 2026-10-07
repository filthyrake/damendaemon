"""Explicit planned-stage control; intermediate activation results are opt-in."""
import torch
from .host_stage_channel import HostStageChannel,EXECUTION
from .host_topology import HostTopology,VERSION
from .host_tensor_channel import HostTensorChannel
from .stage_protocol import StageState,DECODE


class PlannedHostStageChannel(HostStageChannel):
    """One coordinator-to-stage link in a separate, identity-bound CPU topology.

    Link ranks remain 0/1. stage_index is the remote global stage (1..N-1).
    The owner must authenticate/pin sockets with host_tls before transport setup.
    No stage execution, routing, provisioning or network recovery occurs here.
    """
    def __init__(self,transport,state,topology,stage_index):
        if (not isinstance(topology,HostTopology) or type(stage_index) is not int
                or not 1<=stage_index<len(topology.ranges)):
            raise ValueError('planned stage link requires a remote stage in a bounded topology')
        if (not isinstance(transport,HostTensorChannel) or not isinstance(state,StageState)
                or transport.identity!=topology.stage_identity(stage_index) or transport.max_frame_bytes!=topology.frame_bytes
                or state.context!=topology.context or state.slots!=topology.slots):
            raise ValueError('planned stage transport/state differs from topology contract')
        super().__init__(transport,state,vocab_size=topology.text_config['vocab_size'])
        self.topology,self.stage_index=topology,stage_index
        self.result_kind=topology.result_kind(stage_index)

    def handshake(self):
        with self.lock:
            try:
                super().handshake()
                start,end=self.topology.ranges[self.stage_index]
                expected=[VERSION,self.stage_index,len(self.topology.ranges),start,end,self.topology.width,
                          1 if self.result_kind=='activation' else 2]
                if self._exchange(expected)!=expected:raise ValueError('planned host stage contract disagrees')
                with self.close_lock:
                    if self.closed:raise RuntimeError('planned host stage closed during setup')
            except BaseException:self.abort();raise

    def begin(self,command=None):
        with self.lock:
            try:
                result=super().begin(command)
                if result.operation in EXECUTION and (result.rows*self.topology.width*4>self.topology.frame_bytes
                        or (result.operation!=DECODE and result.rows>self.topology.chunk_size)):
                    raise ValueError('planned stage command exceeds activation/chunk budget')
                return result
            except BaseException:self.abort();raise

    def _valid_tokens(self,tokens):
        return all(0<=token<self.topology.mapped_vocab_size and token not in self.topology.excluded_ids
                   for token in tokens.tolist())

    def complete_tokens(self,command,tokens=None):
        with self.lock:
            if self.result_kind!='tokens':
                self.abort();raise ValueError('intermediate stage requires activation completion')
            return super().complete_tokens(command,tokens)

    def complete_activation(self,command,activation=None):
        with self.lock:
            try:
                self._pending(command)
                if self.result_kind!='activation' or command.operation not in EXECUTION:
                    raise ValueError('stage command has no intermediate activation result')
                shape=(command.rows,self.topology.width)
                if command.rows*self.topology.width*4>self.transport.max_frame_bytes:
                    raise ValueError('intermediate activation exceeds frame budget')
                if self.rank==1:
                    if (not isinstance(activation,torch.Tensor) or activation.device.type!='cpu'
                            or activation.dtype!=torch.float32 or tuple(activation.shape)!=shape
                            or not activation.is_contiguous() or not torch.isfinite(activation).all()):
                        raise ValueError('invalid CPU stage activation result')
                    self.transport.send(activation)
                else:
                    if activation is not None:raise ValueError('coordinator cannot supply remote activation results')
                    activation=self.transport.receive(shape,torch.float32)
                    if not torch.isfinite(activation).all():raise ValueError('nonfinite CPU stage activation result')
                self._commit(command)
                return activation
            except BaseException:self.abort();raise
