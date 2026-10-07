"""Greedy completion streams over an exclusively owned prepared host session."""
from contextlib import closing
import threading

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
                    value=worker.prefill_chunk(tokens[start:start+chunk_size],slot,start)
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
