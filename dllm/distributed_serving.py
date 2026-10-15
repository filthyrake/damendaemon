"""Single-owner resident distributed execution behind the existing HTTP API.

HTTP threads validate and queue work; only the owning thread touches the GPU
pair. Final results are published only after both ranks release the slot.
"""
from dataclasses import dataclass, field
import queue
import threading
import time

from .prompts import encode, decode_output
from .prefill import groups
from .runtime import BusyError
from .sampling import Sampling
from .stopping import text_delta
from .request_scope import cache_scope
from .distributed_admission import DistributedAdmission
from .client_decode_turns import ClientDecodeTurns
from .client_prefill_turns import ClientPrefillTurns

STREAM_BUFFER_EVENTS=8
_pair_claim_lock=threading.Lock()


@dataclass
class Request:
    tokens: list
    maximum: int
    submitted: float
    events: queue.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    cache_scope: object = field(default_factory=cache_scope.get,repr=False)


@dataclass
class Active:
    request: Request
    slot: int
    started: float
    output: list = field(default_factory=list)
    emitted: str = ''
    first: float = 0
    generated: float = 0
    done: bool = False
    finish: str = 'length'
    batch_size: int = 1
    prefill_batch_size: int = 1
    acquired: bool = False
    prefill_offset: int = 0
    prefill_turn: int = 0


class DistributedRuntime:
    source_tier='gpu'
    kernel_backend='distributed-stage-graphs'

    def __init__(self, pair, alias='model', queue_size=8, queue_timeout=120, *, client_decode_turns=False,interleaved_prefill=False,client_prefill_turns=False):
        if type(client_decode_turns) is not bool:raise ValueError('client decode turns must be boolean')
        if type(client_prefill_turns) is not bool or (client_prefill_turns and not interleaved_prefill):
            raise ValueError('client prefill turns require explicit interleaved prefill')
        if type(interleaved_prefill) is not bool:raise ValueError('interleaved prefill must be boolean')
        if interleaved_prefill and (not callable(getattr(pair,'prefill_chunk',None))
                or type(getattr(pair,'chunk_size',None)) is not int or not 1<=pair.chunk_size<=4096):
            raise ValueError('interleaved prefill requires an explicit bounded chunk-capable pair')
        if pair.rank != 0:raise ValueError('only the coordinator exposes HTTP')
        if not isinstance(alias,str) or not 1 <= len(alias) <= 128:raise ValueError('invalid model alias')
        if type(queue_size) is not int or not 1 <= queue_size <= 64:raise ValueError('queue size must be 1..64')
        if not 0 < queue_timeout <= 300:raise ValueError('queue timeout must be within 300 seconds')
        self.pair,self.alias,self.queue_timeout=pair,alias,queue_timeout
        self.requests=DistributedAdmission(queue_size,queue_timeout)
        self.closed=threading.Event()
        self.prepare_lock=threading.Lock()
        self.admission_lock=threading.RLock()
        self.draining=False;self.accepted={}
        self.failure=None
        self.active=False
        self.max_batch=getattr(pair,'max_batch',1)
        self.packed_prefill=getattr(pair,'packed_prefill',False)
        self.completed=0
        self.heartbeats=0
        self.client_decode_turns=client_decode_turns;self.decode_turns=ClientDecodeTurns()
        self.interleaved_prefill=interleaved_prefill;self.prefill_turn=0
        self.client_prefill_turns=client_prefill_turns;self.prefill_turns=ClientPrefillTurns()
        self.execution_lock=threading.Lock();self.owner=object()
        with _pair_claim_lock:
            state=getattr(pair,'state',None)
            if (getattr(pair,'_serving_owner',None) is not None or state is None
                    or getattr(state,'failed',False) or getattr(state,'closed',False)
                    or getattr(state,'pending',None) is not None or any(state.lengths)):
                raise ValueError('distributed serving requires an unclaimed ready empty pair')
            pair._serving_owner=self.owner

    def scheduler_status(self):
        """Admission ownership only; never wait for transport or tokenization."""
        with self.admission_lock:
            pending=list(self.accepted.values())
            queued=[request for request in pending if not request.started.is_set()]
            return dict(closed=self.closed.is_set(),failed=self.failure is not None,draining=self.draining,
                drained=self.draining and not pending and not self.active and not self.closed.is_set() and self.failure is None,
                worker_busy=self.active,queue_capacity=self.requests.maxsize,queue_timeout_seconds=self.queue_timeout,
                batch_capacity=self.max_batch,queued_generation_requests=len(queued),
                worker_owned_generation_requests=len(pending)-len(queued),queued_commands=0,worker_owned_commands=0,
                oldest_queued_seconds=max((max(0,time.monotonic()-r.submitted) for r in queued),default=0))

    def drain(self):
        with self.admission_lock:
            self.draining=True
            return self.scheduler_status()

    def _check_admission(self):
        if self.closed.is_set():raise BusyError('distributed session is closed')
        if self.draining:raise BusyError('distributed session is draining')

    def _retire(self,requests):
        with self.admission_lock:
            for request in requests:self.accepted.pop(id(request),None)

    def _close_admission(self):
        with self.admission_lock:
            self.closed.set()
            while True:
                try:waiting=self.requests.get_nowait()
                except queue.Empty:break
                self._publish(waiting,BusyError('distributed session is closed'))
                self.accepted.pop(id(waiting),None)

    def model_ids(self):return [self.alias]

    def manage(self, action, **options):
        if action!='status' or options:raise ValueError('distributed model management is not supported')
        return dict(device='distributed-cuda',context=self.pair.model.max_context,
            models=[dict(id=self.alias,residency='gpu',quantization=self.pair.model.quantization,
                         model_sha256=self.pair.identity.hex())],
            weight_layout='q4-adjacent-body' if getattr(self.pair,'q4_stage_adjacent',False) else 'canonical',
            distributed=True,stage_layers=[[0,12],[12,24]],max_batch_size=self.max_batch,packed_prefill=self.packed_prefill,
            active=self.active,queued=self.requests.qsize(),completed=self.completed,
            heartbeats=self.heartbeats,stream_buffer_events=STREAM_BUFFER_EVENTS,client_admission_turns=True,
            client_decode_turns=self.client_decode_turns,client_prefill_turns=self.client_prefill_turns,
            interleaved_prefill=self.interleaved_prefill,
            failed=self.failure is not None,closed=self.closed.is_set())

    def embed(self,*args,**kwargs):raise ValueError('distributed embeddings are unsupported')
    def score(self,*args,**kwargs):raise ValueError('distributed token scoring is unsupported')
    def rerank(self,*args,**kwargs):raise ValueError('distributed reranking is unsupported')

    def generate(self, alias, messages, max_tokens, *, raw_prompt=None, enable_thinking=None, **options):
        if alias!=self.alias:raise ValueError('unknown distributed model')
        with self.admission_lock:self._check_admission()
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.pair.model.max_context:
            raise ValueError('invalid completion limit')
        stops=options.pop('stop',None)
        if stops or set(options)-set(Sampling.__dataclass_fields__):
            raise ValueError('distributed serving supports plain greedy generation only')
        if Sampling(**options)!=Sampling():
            raise ValueError('distributed serving supports default greedy sampling only')
        with self.prepare_lock:
            tokens=encode(self.pair.model,messages,raw_prompt=raw_prompt,enable_thinking=enable_thinking)
        if not tokens or len(tokens)+max_tokens>self.pair.model.max_context:
            raise ValueError('prompt plus completion limit exceeds distributed context')
        # One extra position is reserved for a release-acknowledged final or error.
        # The owner never schedules a token when eight data events are buffered.
        request=Request(tokens,max_tokens,time.monotonic(),queue.Queue(STREAM_BUFFER_EVENTS+1))
        with self.admission_lock:
            self._check_admission()
            try:self.requests.put_nowait(request)
            except queue.Full:raise BusyError('distributed request queue is full') from None
            self.accepted[id(request)]=request
        received=False
        try:
            while True:
                try:event=request.events.get(timeout=.1)
                except queue.Empty:
                    if self.closed.is_set():raise BusyError('distributed session is closed')
                    if not received and not request.started.is_set() and time.monotonic()-request.submitted>=self.queue_timeout:
                        raise BusyError('distributed request queue deadline exceeded')
                    continue
                if isinstance(event,Exception):raise event
                received=True
                yield event
                if 'finish_reason' in event:return
        finally:
            request.cancelled.set()
            with self.admission_lock:
                if self.requests.discard_cancelled(request):
                    self.accepted.pop(id(request),None)

    @staticmethod
    def _publish(request,event):
        if not request.cancelled.is_set():request.events.put_nowait(event)

    def _advance(self, active, token, batch_size):
        request=active.request
        if request.cancelled.is_set() or self.closed.is_set():return
        now=time.monotonic()
        if not active.output:active.first=now
        active.output.append(token)
        active.batch_size=max(active.batch_size,batch_size)
        stop=token in self.pair.model.eos
        active.done=stop or len(active.output)==request.maximum
        active.finish='stop' if stop else 'length'
        delta,active.emitted,_=text_delta(decode_output(self.pair.model,active.output),
            active.emitted,final=active.done)
        self._publish(request,dict(text=delta))
        active.generated=time.monotonic()

    def _release(self, active):
        # Cancellation can arrive after admission but before prefill dispatch.
        # Such a request never owned a peer slot and must not send RELEASE.
        active=[item for item in active if item.acquired]
        if not active:return
        started=time.monotonic()
        if self.max_batch==1:self.pair.release()
        else:self.pair.release([item.slot for item in active])
        released=time.monotonic()
        for item in active:
            request=item.request
            if request.cancelled.is_set() or self.closed.is_set():continue
            count=len(item.output)
            self._publish(request,dict(finish_reason=item.finish,
                usage=dict(prompt_tokens=len(request.tokens),completion_tokens=count,
                           total_tokens=len(request.tokens)+count),
                timing=dict(source_tier=self.source_tier,activation_seconds=0,
                    queue_seconds=item.started-request.submitted,
                    first_token_seconds=item.first-request.submitted,
                    generation_first_token_seconds=item.first-item.started,
                    generation_seconds=item.generated-item.started,
                    release_seconds=released-started,total_seconds=released-request.submitted,
                    decode_tokens_per_second=(count-1)/(item.generated-item.first) if count>1 else None,
                    max_batch_size=item.batch_size,prefill_batch_size=item.prefill_batch_size,kernel_backend=self.kernel_backend,
                    weight_quantization=self.pair.model.quantization)))
            self.completed+=1

    def _prefill_one_chunk(self,active):
        pending=[item for item in active if not item.output and not item.request.cancelled.is_set()]
        if not pending or self.closed.is_set():self.prefill_turns.reset();return False
        if self.client_prefill_turns:
            item=self.prefill_turns.choose([(item,item.request,min(self.pair.chunk_size,len(item.request.tokens)-item.prefill_offset))
                for item in pending])[0]
        else:item=min(pending,key=lambda item:item.prefill_turn)
        start=item.prefill_offset;end=min(len(item.request.tokens),start+self.pair.chunk_size)
        item.acquired=True
        token=self.pair.prefill_chunk(item.request.tokens[start:end],item.slot,start)
        item.prefill_offset=end
        self.prefill_turn+=1;item.prefill_turn=self.prefill_turn
        # Intermediate chunk samples are protocol results, never output tokens.
        if end==len(item.request.tokens):self._advance(item,token,1)
        if not any(not other.output and not other.request.cancelled.is_set() for other in active):self.prefill_turns.reset()
        return True

    def run(self, idle_timeout=0, heartbeat_interval=5, *, stop_after_drain=False):
        if type(stop_after_drain) is not bool:raise ValueError('stop after drain must be boolean')
        if not 0 < heartbeat_interval <= 10 or not 0 <= idle_timeout <= 86400:
            raise ValueError('invalid distributed serving time bounds')
        if not self.execution_lock.acquire(blocking=False):raise BusyError('distributed execution already has an owner')
        try:
            if self.closed.is_set():raise BusyError('distributed runtime is closed')
            try:return self._run_owned(idle_timeout,heartbeat_interval,stop_after_drain)
            except BaseException as primary:
                self._execution_failed(primary)
                raise
        finally:self.execution_lock.release()

    def _execution_failed(self,primary):
        """Backend-specific teardown, only after actual owned execution fails."""

    def _run_owned(self,idle_timeout,heartbeat_interval,stop_after_drain):
        last_request=last_control=time.monotonic()
        active=[]
        try:
            while not self.closed.is_set():
                retired=[item for item in active if item.done or item.request.cancelled.is_set()]
                progressed=bool(retired)
                self._release(retired)
                self._retire(item.request for item in retired)
                active=[item for item in active if item not in retired]
                self.active=bool(active)
                free=[slot for slot in range(self.max_batch) if all(item.slot!=slot for item in active)]
                # Admit at most one prompt per free slot before advancing decode.
                # Short/EOS requests cannot monopolize the owner by refilling here.
                pending=[]
                for slot in free:
                    try:request=self.requests.get(timeout=.1) if not active else self.requests.get_nowait()
                    except queue.Empty:break
                    if request.cancelled.is_set() or time.monotonic()-request.submitted>=self.queue_timeout:
                        self._publish(request,BusyError('distributed request queue deadline exceeded'))
                        self._retire([request])
                        continue
                    request.started.set()
                    item=Active(request,slot,time.monotonic())
                    active.append(item);self.active=True
                    progressed=True
                    pending.append(item)
                if self.interleaved_prefill:
                    if self._prefill_one_chunk(active):
                        progressed=True;last_request=last_control=time.monotonic()
                else:
                    batches=(groups([item.request.tokens for item in pending],pending,self.pair.model.max_context)
                             if self.packed_prefill else (([item.request.tokens],[item]) for item in pending))
                    for _,items in batches:
                        if self.closed.is_set():break
                        items=[item for item in items if not item.request.cancelled.is_set()]
                        if not items:continue
                        for item in items:item.acquired=True
                        if len(items)>1:
                            result=self.pair.prefill_batch([item.request.tokens for item in items],[item.slot for item in items])
                        else:
                            item=items[0]
                            result=[self.pair.step(item.request.tokens) if self.max_batch==1 else
                                    self.pair.prefill(item.request.tokens,item.slot)]
                        if len(result)!=len(items):raise RuntimeError('distributed prefill result size mismatch')
                        progressed=True
                        for item,token in zip(items,result):
                            item.prefill_batch_size=len(items)
                            self._advance(item,token,1)
                        last_request=last_control=time.monotonic()
                ready=[item for item in active if item.output and not item.done and not item.request.cancelled.is_set()
                       and item.request.events.qsize()<STREAM_BUFFER_EVENTS]
                if self.client_decode_turns:
                    ready=[item for _,_,item in self.decode_turns.choose([(item.slot,item.request,item) for item in ready])]
                if not active:self.decode_turns.reset();self.prefill_turn=0;self.prefill_turns.reset()
                if ready and not self.closed.is_set():
                    tokens=[item.output[-1] for item in ready]
                    result=([self.pair.step(tokens)] if self.max_batch==1 else
                            self.pair.decode(tokens,[item.slot for item in ready]))
                    if len(result)!=len(ready):raise RuntimeError('distributed decode result size mismatch')
                    progressed=True
                    for item,token in zip(ready,result):self._advance(item,token,len(ready))
                    last_request=last_control=time.monotonic()
                now=time.monotonic()
                if stop_after_drain and self.scheduler_status()['drained']:break
                if not active and idle_timeout and now-last_request>=idle_timeout:break
                # A live slot can be idle because its consumer is backpressured.
                # Keep the peer alive without advancing its state or busy spinning.
                if now-last_control>=heartbeat_interval:
                    self.pair.heartbeat();self.heartbeats+=1;last_control=time.monotonic()
                if active and not progressed:self.closed.wait(min(.01,heartbeat_interval))
        except BaseException as error:
            self.failure=error
            for item in active:self._publish(item.request,RuntimeError('distributed worker session failed'))
            raise
        finally:
            self._close_admission()
            try:
                if active and not self.pair.state.failed:self._release(active)
            except BaseException as cleanup:
                if self.failure is None:self.failure=cleanup;raise
                self.failure.add_note('distributed slot release failed; retain pair for backend cleanup')
            finally:
                self.active=False
                self.decode_turns.reset();self.prefill_turn=0;self.prefill_turns.reset()
                with self.admission_lock:self.accepted.clear()

    def close(self):self._close_admission()
