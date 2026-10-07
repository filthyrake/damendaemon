"""Prepared CPU hybrid layer workers over explicitly owned trusted sockets.

No listener, remote provisioning, GPU execution or serving registration. Weight
identity is supplied to the transport by its owner; setup additionally checks
configuration, precision and contiguous layer ownership. Only sampled IDs cross
back to the coordinator. Failed epochs are discarded, never replayed.
"""
import hashlib
import json
import threading
import torch
from .host_stage_channel import HostStageChannel
from .hybrid_stage import HybridLayerStage
from .stage_protocol import Command,PREFILL,PREFILL_BATCH,PREFILL_CHUNK,DECODE,RELEASE,PING,STOP


class HostHybridWorker:
    def __init__(self,channel,stage):
        if (not isinstance(channel,HostStageChannel) or channel.ready or channel.closed
                or getattr(channel,'_worker_owner',None) is not None
                or not isinstance(stage,HybridLayerStage) or stage.closed or getattr(stage,'failed',False)
                or getattr(stage,'_pipeline_owner',None) is not None or any(stage.lengths)
                or stage.actual_device.type!='cpu' or stage.compute_dtype!=torch.float32
                or stage.context!=channel.state.context or stage.max_batch!=channel.state.slots
                or stage.c['vocab_size']!=channel.vocab_size
                or (channel.rank==0 and (not stage.first or stage.last))
                or (channel.rank==1 and (stage.first or not stage.last))):
            raise ValueError('host hybrid worker needs fresh compatible CPU FP32 stage ownership')
        encoded=json.dumps(dict(config=stage.c,dtype=str(stage.compute_dtype),quantization=stage.model.quantization),sort_keys=True,
                           separators=(',',':'),allow_nan=False).encode()
        if len(encoded)>1024**2:raise ValueError('host hybrid configuration exceeds metadata limit')
        self.contract=hashlib.sha256(encoded).digest()
        self.channel,self.stage,self.rank=channel,stage,channel.rank
        self.width=stage.c['hidden_size'];self.ready=self.failed=self.closed=False
        self.lock=threading.RLock();self.owner=object()
        channel._worker_owner=stage._pipeline_owner=self.owner

    def close(self):
        # Interrupt blocked transport before waiting for the execution lock.
        self.channel.close()
        with self.lock:
            if self.closed:return
            self.ready=False
            if not self.channel.state.closed:self.failed=True
            self.stage.close() # A failed close retains ownership for retry.
            self.closed=True

    def _fail(self,error):
        self.failed=True;self.ready=False;self.channel.abort()
        try:self.close()
        except BaseException:error.add_note('host hybrid stage cleanup failed; owner retained for retry')

    def _exchange_setup(self,tensor):
        transport=self.channel.transport
        if self.rank==0:
            transport.send(tensor);return transport.receive(tensor.shape,tensor.dtype)
        peer=transport.receive(tensor.shape,tensor.dtype);transport.send(tensor);return peer

    def handshake(self):
        with self.lock:
            try:
                self.channel.handshake()
                s=self.stage
                record=[s.layer_start,s.layer_end,s.c['num_hidden_layers'],self.width,s.context,s.max_batch]
                peer=self._exchange_setup(torch.tensor(record,dtype=torch.int64,device='cpu')).tolist()
                if (peer[2:]!=record[2:] or (self.rank==0 and peer[:2]!=[s.layer_end,record[2]])
                        or (self.rank==1 and peer[:2]!=[0,s.layer_start])):
                    raise ValueError('host hybrid layer coverage disagrees')
                identity=torch.tensor(list(self.contract),dtype=torch.uint8,device='cpu')
                if not torch.equal(self._exchange_setup(identity),identity):
                    raise ValueError('host hybrid model configuration/precision disagrees')
                self.ready=True
            except BaseException as error:self._fail(error);raise

    def _ready(self,coordinator=False):
        if not self.ready or self.failed or self.closed:raise RuntimeError('host hybrid worker is not ready')
        if coordinator and self.rank!=0:raise ValueError('only coordinator submits work')

    def _tokens(self,tokens):
        if (not isinstance(tokens,(list,tuple)) or not tokens or len(tokens)>self.channel.state.context
                or any(type(v) is not int or not 0<=v<self.channel.vocab_size for v in tokens)
                or set(tokens)&set(self.stage.c.get('multimodal_token_ids',[]))):
            raise ValueError('invalid host hybrid text token IDs')

    def _slots(self,slots):
        if (not isinstance(slots,(list,tuple)) or not slots or len(slots)>self.stage.max_batch
                or any(type(s) is not int or not 0<=s<self.stage.max_batch for s in slots)
                or len(set(slots))!=len(slots)):
            raise ValueError('invalid host hybrid slot selection')

    def _command(self,operation,rows,entries):
        if rows*self.width*4>self.channel.transport.max_frame_bytes:
            raise ValueError('host hybrid activation exceeds negotiated frame cap')
        state=self.channel.state
        return Command(state.epoch,state.sequence,operation,rows,tuple(entries))

    def prefill(self,tokens,slot=0):return self.prefill_batch([tokens],[slot])[0]

    def prefill_batch(self,prompts,slots):
        with self.lock:
            self._ready(True);self._slots(slots)
            if not isinstance(prompts,(list,tuple)) or len(prompts)!=len(slots):raise ValueError('invalid host hybrid prompt batch')
            for tokens in prompts:self._tokens(tokens)
            state=self.channel.state;rows=sum(map(len,prompts))
            if rows>state.context or any(state.lengths[s] for s in slots):raise ValueError('occupied slot or prompt budget exceeded')
            operation=PREFILL if len(prompts)==1 else PREFILL_BATCH
            entries=[(slot,state.generations[slot]+1,0 if operation==PREFILL else len(prompt)) for slot,prompt in zip(slots,prompts)]
            return self._execute(self._command(operation,rows,entries),[v for prompt in prompts for v in prompt])

    def prefill_chunk(self,tokens,slot=0,start=0):
        """Return this chunk's sampled continuation; callers choose when to decode."""
        with self.lock:
            self._ready(True);self._slots([slot]);self._tokens(tokens)
            state=self.channel.state
            if (type(start) is not int or start<0 or start+len(tokens)>state.context
                    or (start==0 and state.lengths[slot])
                    or (start>0 and start!=state.lengths[slot])):
                raise ValueError('invalid host hybrid prefill chunk offset/ownership')
            generation=state.generations[slot]+(start==0)
            command=self._command(PREFILL_CHUNK,len(tokens),[(slot,generation,start)])
            return self._execute(command,list(tokens))[0]

    def decode(self,tokens,slots):
        with self.lock:
            self._ready(True);self._tokens(tokens);self._slots(slots)
            state=self.channel.state
            if len(tokens)!=len(slots) or any(not 0<state.lengths[s]<state.context for s in slots):
                raise ValueError('invalid host hybrid decode ownership/context')
            command=self._command(DECODE,len(tokens),[(s,state.generations[s],state.lengths[s]) for s in slots])
            return self._execute(command,list(tokens))

    def release(self,slots):
        with self.lock:
            self._ready(True);self._slots(slots);state=self.channel.state
            if any(not state.lengths[s] for s in slots):raise ValueError('cannot release an empty host hybrid slot')
            return self._execute(self._command(RELEASE,0,[(s,state.generations[s],state.lengths[s]) for s in slots]))

    def ping(self):
        with self.lock:self._ready(True);return self._execute(self._command(PING,0,()))

    def stop(self):
        with self.lock:self._ready(True);return self._execute(self._command(STOP,0,()))

    def step(self):
        with self.lock:
            self._ready()
            if self.rank!=1:raise ValueError('only output worker receives commands')
            return self._execute()

    def _run(self,command,values):
        stage=self.stage
        if command.operation==DECODE:
            ids=torch.tensor([entry[0] for entry in command.entries],dtype=torch.int64,device='cpu')
            positions=torch.tensor([[entry[2]] for entry in command.entries],dtype=torch.int64,device='cpu')
            hidden=stage.embedding(values[:,None]) if self.rank==0 else values[:,None,:]
            result=stage.forward_layers(hidden,positions,ids,False).squeeze(1)
            return result,result
        chunks=[];last=[];offset=0
        for slot,_,extent in command.entries:
            size=extent if command.operation==PREFILL_BATCH else command.rows
            start=extent if command.operation==PREFILL_CHUNK else 0
            chunk=values[offset:offset+size]
            hidden=stage.embedding(chunk[None]) if self.rank==0 else chunk[None]
            result=stage.forward_layers(hidden,torch.arange(start,start+size,device='cpu')[None],
                                        torch.tensor([slot],dtype=torch.int64,device='cpu'),True,start)[0]
            chunks.append(result);last.append(result[-1]);offset+=size
        return torch.cat(chunks),torch.stack(last)

    def _clear_slots(self,command):
        for slot,_,_ in command.entries:
            for tensor in self.stage.kv.values():tensor[:,slot].zero_()
            for group in (self.stage.conv,self.stage.state):
                for tensor in group.values():tensor[slot].zero_()

    @torch.inference_mode()
    def _execute(self,command=None,tokens=None):
        try:
            command=self.channel.begin(command)
            if command.operation in (RELEASE,PING,STOP):
                if command.operation==RELEASE:self._clear_slots(command)
                if command.operation==STOP:self.stage.close()
                self.channel.finish(command)
                self.stage.lengths[:]=self.channel.state.lengths;self.stage.length=self.stage.lengths[0]
                if command.operation==STOP:self.closed=True;self.ready=False
                return None
            shape=(command.rows,self.width)
            if self.rank==0:
                values=torch.tensor(tokens,dtype=torch.int64,device='cpu')
                hidden,_=self._run(command,values)
                if hidden.shape!=shape or hidden.dtype!=torch.float32 or not torch.isfinite(hidden).all():
                    raise ValueError('invalid host hybrid intermediate activation')
                self.channel.transport.send(hidden.contiguous())
                result=self.channel.complete_tokens(command)
            else:
                hidden=self.channel.transport.receive(shape,torch.float32)
                if not torch.isfinite(hidden).all():raise ValueError('nonfinite host hybrid received activation')
                _,last=self._run(command,hidden)
                logits=self.stage.output_logits(last)
                if logits.shape!=(len(command.entries),self.channel.vocab_size) or not torch.isfinite(logits).all():
                    raise ValueError('invalid host hybrid output logits')
                result=self.channel.complete_tokens(command,logits.argmax(-1))
            self.stage.lengths[:]=self.channel.state.lengths;self.stage.length=self.stage.lengths[0]
            return result.tolist() if self.rank==0 else None
        except BaseException as error:self._fail(error);raise
