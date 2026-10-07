"""Explicit prepared CPU stages behind a synchronous authenticated star topology."""
import threading
import torch

from .cpu_hybrid_stage import CpuHybridStage
from .glm_stage_slots import GlmStageSlots
from .glm_flash_stage_slots import FlashStageSlots
from .deepseek_stage_slots import DeepSeekStageSlots
from .host_hybrid_worker import HostHybridWorker
from .host_topology import HostTopology
from .planned_host_stage_channel import PlannedHostStageChannel
from .stage_protocol import DECODE,RELEASE,PING,STOP,PREFILL_BATCH,PREFILL_CHUNK

_claim_lock=threading.Lock()


class _PlannedWorker(HostHybridWorker):
    def _initialize(self,stage,topology,index,channels,rank):
        with _claim_lock:
            if (not isinstance(topology,HostTopology) or type(index) is not int or not 0<=index<len(topology.ranges)):
                raise ValueError('prepared host pipeline requires an explicit topology and valid stage index')
            expected=list(range(1,len(topology.ranges))) if rank==0 else [index]
            if (not isinstance(channels,(list,tuple)) or len(channels)!=len(expected)
                    or any(not isinstance(c,PlannedHostStageChannel) for c in channels)
                    or [c.stage_index for c in channels]!=expected
                    or any(c.rank!=rank or c.topology.identity!=topology.identity or c.ready or c.closed
                           or getattr(c,'_worker_owner',None) is not None or c.state.failed or c.state.closed
                           or c.state.pending is not None or c.state.sequence!=1 or any(c.state.lengths) or any(c.state.generations)
                           or c.transport.ready or c.transport.closed or c.transport.failed
                           for c in channels)
                    or len({c.state.epoch for c in channels})!=1):
                raise ValueError('host pipeline needs fresh ordered links matching one topology and epoch')
            glm=topology.config.get('model_type')=='glm_moe_dsa'
            flash=topology.config.get('model_type')=='glm5_next'
            deepseek=topology.routing_tokens
            reference=glm or flash or deepseek
            expected_stage=DeepSeekStageSlots if deepseek else FlashStageSlots if flash else GlmStageSlots if glm else CpuHybridStage
            if (not isinstance(stage,expected_stage) or stage.closed or getattr(stage,'failed',False)
                    or getattr(stage,'_pipeline_owner',None) is not None or any(stage.lengths)
                    or stage.actual_device.type!='cpu' or stage.compute_dtype!=torch.float32
                    or (stage.layer_start,stage.layer_end)!=topology.ranges[index]
                    or stage.context!=topology.context or stage.max_batch!=topology.slots
                    or stage.c!=topology.config
                    or (reference and (any(state is not None for state in stage.states) or (stage.stage.prefill_cleanup_pending if deepseek else stage.stage.failed_cleanups)))
                    or (not reference and stage.model.quantization!=topology.quantization)):
                raise ValueError('prepared CPU stage differs from topology or already has an owner')
            self.topology,self.stage_index,self.channels=topology,index,tuple(channels)
            self.channel=self.channels[-1];self.stage,self.rank=stage,rank
            self.width=topology.width;self.contract=topology.identity
            self.mapped_vocab_size=topology.mapped_vocab_size;self.excluded_ids=frozenset(topology.excluded_ids)
            self.ready=self.failed=self.closed=False;self.lock=threading.RLock();self.owner=object()
            stage._pipeline_owner=self.owner
            for channel in self.channels:channel._worker_owner=self.owner

    def handshake(self):
        with self.lock:
            try:
                for channel in self.channels:channel.handshake()
                self.ready=True
            except BaseException as error:self._fail(error);raise

    def close(self):
        # Interrupt every link before waiting for the model-operation lock.
        errors=[]
        for channel in self.channels:
            try:channel.close()
            except BaseException as error:errors.append(error)
        with self.lock:
            self.ready=False
            if any(not c.state.closed for c in self.channels):self.failed=True
            try:self.stage.close()
            except BaseException as error:errors.append(error)
            if not errors:self.closed=True
        if errors:
            if len(errors)>1:errors[0].add_note('additional host pipeline owners failed cleanup; retain worker and retry close')
            raise errors[0]

    def _fail(self,error):
        self.failed=True;self.ready=False
        try:self.close()
        except BaseException:error.add_note('host pipeline cleanup failed; retain worker and retry close')

    def _command(self,operation,rows,entries):
        if operation not in (DECODE,RELEASE,PING,STOP) and rows>self.topology.chunk_size:
            raise ValueError('host pipeline prefill exceeds explicit chunk rows')
        return super()._command(operation,rows,entries)

    def _run(self,command,values,routing=None):
        deepseek=isinstance(self.stage,DeepSeekStageSlots)
        if not isinstance(self.stage,(FlashStageSlots,DeepSeekStageSlots)):return super()._run(command,values)
        if deepseek:routing=self._routing(values if self.rank==0 else routing,command.rows)
        stage=self.stage;streams,dim=stage.stage.streams,stage.stage.dim
        if command.operation==DECODE:
            ids=torch.tensor([entry[0] for entry in command.entries],dtype=torch.int64)
            positions=torch.tensor([[entry[2]] for entry in command.entries],dtype=torch.int64)
            hidden=stage.embedding(values[:,None]) if self.rank==0 else values.reshape(-1,1,streams,dim)
            result=stage.forward_layers(hidden,positions,ids,False,**(dict(token_ids=routing[:,None]) if deepseek else {})).squeeze(1)
            return result.flatten(1),result
        chunks=[];last=[];offset=0
        for slot,_,extent in command.entries:
            size=extent if command.operation==PREFILL_BATCH else command.rows
            start=extent if command.operation==PREFILL_CHUNK else 0
            chunk=values[offset:offset+size]
            hidden=stage.embedding(chunk[None]) if self.rank==0 else chunk.reshape(1,size,streams,dim)
            result=stage.forward_layers(hidden,torch.arange(start,start+size)[None],torch.tensor([slot]),True,start,
                **(dict(token_ids=routing[offset:offset+size][None]) if deepseek else {}))[0]
            chunks.append(result.flatten(1));last.append(result[-1]);offset+=size
        return torch.cat(chunks),torch.stack(last)

    def _clear_slots(self,command):
        if isinstance(self.stage,(GlmStageSlots,FlashStageSlots,DeepSeekStageSlots)):
            self.stage.release([slot for slot,_,_ in command.entries])
        else:super()._clear_slots(command)

    def _commit_stage(self):
        self.stage.lengths[:]=self.channel.state.lengths;self.stage.length=self.stage.lengths[0]

    def _routing(self,tokens,rows):
        if (not isinstance(tokens,torch.Tensor) or tokens.device.type!='cpu' or tokens.dtype!=torch.int64
                or tuple(tokens.shape)!=(rows,) or not tokens.is_contiguous()
                or rows*8>self.topology.frame_bytes or not self.channel._valid_tokens(tokens)):
            raise ValueError('invalid DeepSeek host routing token frame')
        return tokens

    def _activation(self,values,rows):
        if (not isinstance(values,torch.Tensor) or tuple(values.shape)!=(rows,self.width)
                or values.dtype!=torch.float32 or values.device.type!='cpu' or not torch.isfinite(values).all()):
            raise ValueError('invalid CPU host pipeline activation')
        return values.contiguous()


class HostPipelineCoordinator(_PlannedWorker):
    """Own first-stage execution and one explicitly prepared link per remote range."""
    def __init__(self,stage,topology,channels):self._initialize(stage,topology,0,channels,0)

    @torch.inference_mode()
    def prefill_embeddings(self,values,slot=0,start=0):
        """Consume one owned CPU FP32 [sequence, hidden] prepared Flash chunk.

        This low-level API validates storage and cursors, not image provenance.
        Token decode and output exclusions remain the topology's existing policy.
        """
        with self.lock:
            self._ready(True);self._slots([slot])
            if self.topology.input_kind!='flash_embeddings' or not isinstance(self.stage,FlashStageSlots):
                raise ValueError('prepared embeddings require explicit Flash input mode')
            stage=self.stage.stage
            if (not isinstance(values,torch.Tensor) or values.device.type!='cpu' or values.dtype!=torch.float32
                    or values.ndim!=2 or values.shape[1]!=stage.dim
                    or not 1<=values.shape[0]<=self.topology.chunk_size
                    or values.shape[0]*self.width*4>min(self.topology.frame_bytes,stage.max_output_bytes)
                    or not torch.isfinite(values).all()):
                raise ValueError('invalid bounded Flash prepared embedding chunk')
            command=self._prefill_chunk_command(values.shape[0],slot,start)
            # Detach caller storage before transport admission or cache mutation.
            hidden=values.detach().clone()[None,:,None,:].expand(1,-1,stage.streams,-1)
            return self._execute(command,embeddings=hidden)[0]

    @torch.inference_mode()
    def _execute(self,command=None,tokens=None,*,embeddings=None):
        try:
            # Every participant admits before the first layer mutates local state.
            for channel in self.channels:channel.begin(command)
            if command.operation in (RELEASE,PING,STOP):
                if command.operation==RELEASE:self._clear_slots(command)
                if command.operation==STOP:self.stage.close()
                for channel in self.channels:channel.finish(command)
                self._commit_stage()
                if command.operation==STOP:self.closed=True;self.ready=False
                return None
            if embeddings is None:
                tokens=torch.tensor(tokens,dtype=torch.int64,device='cpu')
                hidden,_=self._run(command,tokens)
            else:
                slot,_,start=command.entries[0]
                hidden=self.stage.forward_layers(embeddings,torch.arange(start,start+command.rows)[None],
                    torch.tensor([slot]),True,start)[0].flatten(1)
            result=None
            for channel in self.channels:
                if self.topology.routing_tokens:channel.transport.send(self._routing(tokens,command.rows))
                channel.transport.send(self._activation(hidden,command.rows))
                if channel.result_kind=='activation':hidden=channel.complete_activation(command)
                else:result=channel.complete_tokens(command)
            self._commit_stage()
            return result.tolist()
        except BaseException as error:self._fail(error);raise


class HostPipelineWorker(_PlannedWorker):
    """Own one intermediate/final range; the coordinator alone submits commands."""
    def __init__(self,stage,channel):
        if not isinstance(channel,PlannedHostStageChannel):raise ValueError('worker requires a planned stage channel')
        self._initialize(stage,channel.topology,channel.stage_index,[channel],1)

    @torch.inference_mode()
    def _execute(self,command=None,tokens=None):
        try:
            command=self.channel.begin()
            if command.operation in (RELEASE,PING,STOP):
                if command.operation==RELEASE:self._clear_slots(command)
                if command.operation==STOP:self.stage.close()
                self.channel.finish(command);self._commit_stage()
                if command.operation==STOP:self.closed=True;self.ready=False
                return None
            routing=(self._routing(self.channel.transport.receive((command.rows,),torch.int64),command.rows)
                     if self.topology.routing_tokens else None)
            hidden=self.channel.transport.receive((command.rows,self.width),torch.float32)
            values,last=self._run(command,self._activation(hidden,command.rows),routing)
            if self.channel.result_kind=='activation':self.channel.complete_activation(command,self._activation(values,command.rows))
            else:
                logits=self.stage.output_logits(last)
                if (logits.shape!=(len(command.entries),self.channel.vocab_size) or logits.dtype!=torch.float32
                        or logits.device.type!='cpu' or not torch.isfinite(logits).all()):
                    raise ValueError('invalid CPU host pipeline output logits')
                scores=logits[:,:self.mapped_vocab_size].clone()
                excluded=[token for token in self.excluded_ids if token<self.mapped_vocab_size]
                if excluded:scores[:,excluded]=-float('inf')
                self.channel.complete_tokens(command,scores.argmax(-1))
            self._commit_stage()
            return None
        except BaseException as error:self._fail(error);raise
