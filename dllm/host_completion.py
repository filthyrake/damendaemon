"""Greedy completion streams over an exclusively owned prepared host session."""
from contextlib import closing
import threading
import torch

from .cpu_pipeline_generation import prepare_cpu_batch
from .host_hybrid_worker import HostHybridWorker
from .hybrid_pipeline import PipelineCancelled


class HostCompletion:
    """Own a ready coordinator; successful requests release slots for reuse.

    The peer owner must continuously call step(). Errors or abandoned iterators
    discard the session. close() interrupts blocked transport and retains this
    owner if stage cleanup must be retried. Cancellation callbacks run between
    commands; use close() from another thread to interrupt an in-flight command.
    Do not submit commands directly to a worker after transferring it here.
    """
    def __init__(self,worker,*,eos_ids=()):
        if not isinstance(worker,HostHybridWorker):raise ValueError('completion requires a host worker')
        with worker.lock:
            if (worker.rank!=0 or not worker.ready or worker.failed or worker.closed
                    or any(worker.channel.state.lengths)
                    or getattr(worker,'_completion_owner',None) is not None):
                raise ValueError('completion requires an unclaimed ready empty coordinator')
            if (not isinstance(eos_ids,(list,tuple,set,frozenset)) or len(eos_ids)>256
                    or any(type(t) is not int or not 0<=t<worker.mapped_vocab_size or t in worker.excluded_ids for t in eos_ids)):
                raise ValueError('invalid host completion EOS policy')
            self.worker=worker
            self.eos_ids=frozenset(eos_ids)
            self.lock=threading.Lock()
            self.stopped=threading.Event()
            worker._completion_owner=object()

    def close(self):
        self.stopped.set()
        self.worker.close()

    cleanup=close

    def _check_cancelled(self,cancelled):
        if self.stopped.is_set() or (cancelled is not None and cancelled()):
            raise PipelineCancelled('host completion cancelled')

    def generate(self,tokens,*,max_new_tokens=128,chunk_size=128,cancelled=None):
        if type(max_new_tokens) is not int:raise ValueError('single host completion requires an integer output limit')
        with closing(self.generate_batch([tokens],max_new_tokens=max_new_tokens,chunk_size=chunk_size,cancelled=cancelled)) as stream:
            for event in stream:yield {k:v for k,v in event.items() if k!='row'}

    def generate_batch(self,rows,*,max_new_tokens=128,chunk_size=128,cancelled=None):
        worker=self.worker
        rows,limits=prepare_cpu_batch(rows,vocab_size=worker.mapped_vocab_size,
            max_context=worker.channel.state.context,max_new_tokens=max_new_tokens,chunk_size=chunk_size,
            eos_ids=self.eos_ids,excluded_ids=tuple(t for t in worker.excluded_ids if t<worker.mapped_vocab_size))
        if (len(rows)>worker.channel.state.slots
                or max(len(rows),min(chunk_size,max(map(len,rows))))*worker.width*4>worker.channel.transport.max_frame_bytes):
            raise ValueError('host completion exceeds session slot/frame capacity')
        topology=getattr(worker,'topology',None)
        if topology is not None and min(chunk_size,max(map(len,rows)))>topology.chunk_size:
            raise ValueError('host completion exceeds planned prefill chunk rows')
        yield from self._generate(rows,limits,chunk_size,cancelled,worker.prefill_chunk)

    def generate_embeddings(self,values,*,max_new_tokens=128,chunk_size=128,cancelled=None):
        """Complete one prepared Flash [sequence, hidden] row on CPU."""
        if type(max_new_tokens) is not int:raise ValueError('single host completion requires an integer output limit')
        with closing(self.generate_embedding_batch([values],max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,cancelled=cancelled)) as stream:
            for event in stream:yield {k:v for k,v in event.items() if k!='row'}

    def generate_embedding_batch(self,rows,*,max_new_tokens=128,chunk_size=128,cancelled=None):
        """Detach bounded prepared features before entering the shared owned loop.

        Features bind storage and positions only; callers own image preparation
        and native prompt policy. Successful rows release their slots for reuse.
        """
        from .host_pipeline_worker import HostPipelineCoordinator
        worker=self.worker;topology=getattr(worker,'topology',None)
        if not isinstance(worker,HostPipelineCoordinator) or topology.input_kind!='flash_embeddings':
            raise ValueError('embedding completion requires explicit Flash prepared-input topology')
        self._check_cancelled(cancelled if callable(cancelled) else None)
        if (not isinstance(rows,(list,tuple)) or not 1<=len(rows)<=topology.slots
                or type(chunk_size) is not int or not 1<=chunk_size<=4096):
            raise ValueError('invalid embedding completion slot/chunk limits')
        limits=[max_new_tokens]*len(rows) if type(max_new_tokens) is int else max_new_tokens
        if (not isinstance(limits,(list,tuple)) or len(limits)!=len(rows)
                or any(type(limit) is not int or not 1<=limit<=4096 for limit in limits)
                or cancelled is not None and not callable(cancelled)):
            raise ValueError('invalid embedding completion output/cancellation limits')
        worker._ready(True)
        dim=topology.text_config['hidden_size']
        output_bytes=getattr(getattr(worker.stage,'stage',None),'max_output_bytes',0)
        retained=0
        for row,limit in zip(rows,limits):
            if (not isinstance(row,torch.Tensor) or row.device.type!='cpu' or row.dtype!=torch.float32
                    or row.ndim!=2 or row.shape[1]!=dim or not 1<=row.shape[0]<=topology.context-limit):
                raise ValueError('invalid embedding completion feature geometry/context')
            retained+=row.numel()*4
            if (retained>output_bytes or min(chunk_size,len(row))>topology.chunk_size
                    or min(chunk_size,len(row))*topology.width*4>min(output_bytes,topology.frame_bytes)
                    or not torch.isfinite(row).all()):
                raise ValueError('embedding completion exceeds storage/frame bounds or has nonfinite features')
        if len(rows)*topology.width*4>min(output_bytes,topology.frame_bytes):
            raise ValueError('embedding completion decode exceeds output/frame bounds')
        prepared=[row.detach().clone() for row in rows]
        yield from self._generate(prepared,list(limits),chunk_size,cancelled,worker.prefill_embeddings)

    def _generate(self,rows,limits,chunk_size,cancelled,prefill):
        worker=self.worker
        if cancelled is not None and not callable(cancelled):raise ValueError('cancellation must be callable')
        if not self.lock.acquire(blocking=False):raise RuntimeError('host completion is active')
        primary=None;counts=[0]*len(rows);reasons=['length']*len(rows)
        try:
            self._check_cancelled(cancelled)
            worker._ready(True)
            if any(worker.channel.state.lengths):raise RuntimeError('host completion session contains occupied slots')
            sampled=[]
            for slot,tokens in enumerate(rows):
                for start in range(0,len(tokens),chunk_size):
                    self._check_cancelled(cancelled)
                    value=prefill(tokens[start:start+chunk_size],slot,start)
                sampled.append(value)
            active=list(range(len(rows)))
            while active:
                self._check_cancelled(cancelled)
                survivors=[];next_tokens=[];finished=[]
                for row,token in zip(active,sampled):
                    self._check_cancelled(cancelled)
                    counts[row]+=1
                    yield dict(row=row,token_id=token)
                    if token in self.eos_ids:
                        reasons[row]='stop';finished.append(row)
                    elif counts[row]>=limits[row]:finished.append(row)
                    else:survivors.append(row);next_tokens.append(token)
                self._check_cancelled(cancelled)
                if finished:worker.release(finished)
                active=survivors
                if active:
                    self._check_cancelled(cancelled)
                    sampled=worker.decode(next_tokens,active)
            self._check_cancelled(cancelled)
        except BaseException as error:
            primary=error
            raise
        finally:
            try:
                if primary is not None:
                    try:self.close()
                    except BaseException as error:
                        primary.add_note(f'host completion cleanup failed; retain owner and retry close: {type(error).__name__}')
            finally:self.lock.release()
        for row,tokens in enumerate(rows):
            yield dict(row=row,finish_reason=reasons[row],usage=dict(prompt_tokens=len(tokens),
                completion_tokens=counts[row],total_tokens=len(tokens)+counts[row]))
