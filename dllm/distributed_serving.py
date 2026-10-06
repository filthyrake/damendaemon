"""Single-owner resident distributed execution behind the existing HTTP API.

HTTP threads validate and queue work; only the owning thread touches the GPU
pair. Final results are published only after both ranks release the slot.
"""
from dataclasses import dataclass, field
import queue
import threading
import time

from .prompts import encode, decode_output
from .runtime import BusyError
from .sampling import Sampling
from .stopping import text_delta


@dataclass
class Request:
    tokens: list
    maximum: int
    submitted: float
    events: queue.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)


class DistributedRuntime:
    def __init__(self, pair, alias='model', queue_size=8, queue_timeout=120):
        if pair.rank != 0:raise ValueError('only the coordinator exposes HTTP')
        if not isinstance(alias,str) or not 1 <= len(alias) <= 128:raise ValueError('invalid model alias')
        if type(queue_size) is not int or not 1 <= queue_size <= 64:raise ValueError('queue size must be 1..64')
        if not 0 < queue_timeout <= 300:raise ValueError('queue timeout must be within 300 seconds')
        self.pair,self.alias,self.queue_timeout=pair,alias,queue_timeout
        self.requests=queue.Queue(queue_size)
        self.closed=threading.Event()
        self.prepare_lock=threading.Lock()
        self.failure=None
        self.active=False
        self.completed=0
        self.heartbeats=0

    def model_ids(self):return [self.alias]

    def manage(self, action, **options):
        if action!='status' or options:raise ValueError('distributed model management is not supported')
        return dict(device='distributed-cuda',context=self.pair.model.max_context,
            models=[dict(id=self.alias,residency='gpu',quantization=self.pair.model.quantization,
                         model_sha256=self.pair.identity.hex())],
            distributed=True,stage_layers=[[0,12],[12,24]],max_batch_size=1,
            active=self.active,queued=self.requests.qsize(),completed=self.completed,
            heartbeats=self.heartbeats,failed=self.failure is not None,closed=self.closed.is_set())

    def embed(self,*args,**kwargs):raise ValueError('distributed embeddings are unsupported')
    def score(self,*args,**kwargs):raise ValueError('distributed token scoring is unsupported')
    def rerank(self,*args,**kwargs):raise ValueError('distributed reranking is unsupported')

    def generate(self, alias, messages, max_tokens, *, raw_prompt=None, enable_thinking=None, **options):
        if alias!=self.alias:raise ValueError('unknown distributed model')
        if self.closed.is_set():raise BusyError('distributed session is closed')
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
        request=Request(tokens,max_tokens,time.monotonic(),queue.Queue(max_tokens+1))
        try:self.requests.put_nowait(request)
        except queue.Full:raise BusyError('distributed request queue is full') from None
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
        finally:request.cancelled.set()

    @staticmethod
    def _publish(request,event):
        if not request.cancelled.is_set():request.events.put_nowait(event)

    def _execute(self,request):
        if request.cancelled.is_set():return
        request.started.set()
        started=time.monotonic();output=[];emitted='';finish='length'
        self.active=True
        try:
            token=self.pair.step(request.tokens)
            first=time.monotonic()
            for index in range(request.maximum):
                if request.cancelled.is_set() or self.closed.is_set():break
                output.append(token)
                stop=token in self.pair.model.eos
                delta,emitted,_=text_delta(decode_output(self.pair.model,output),emitted,
                    final=stop or index==request.maximum-1)
                self._publish(request,dict(text=delta))
                if stop:finish='stop';break
                if index+1<request.maximum:token=self.pair.step([token])
            generated=time.monotonic()
        finally:
            try:
                release_started=time.monotonic()
                if not self.pair.state.failed:self.pair.release()
                released=time.monotonic()
            finally:self.active=False
        if request.cancelled.is_set() or self.closed.is_set():return
        self._publish(request,dict(finish_reason=finish,
            usage=dict(prompt_tokens=len(request.tokens),completion_tokens=len(output),
                       total_tokens=len(request.tokens)+len(output)),
            timing=dict(source_tier='gpu',activation_seconds=0,
                queue_seconds=started-request.submitted,first_token_seconds=first-request.submitted,
                generation_first_token_seconds=first-started,generation_seconds=generated-started,
                release_seconds=released-release_started,total_seconds=released-request.submitted,
                decode_tokens_per_second=(len(output)-1)/(generated-first) if len(output)>1 else None,
                max_batch_size=1,kernel_backend='distributed-stage-graphs',
                weight_quantization=self.pair.model.quantization)))
        self.completed+=1

    def run(self, idle_timeout=0, heartbeat_interval=5):
        if not 0 < heartbeat_interval <= 10 or not 0 <= idle_timeout <= 86400:
            raise ValueError('invalid distributed serving time bounds')
        last_request=last_control=time.monotonic()
        request=None
        try:
            while not self.closed.is_set():
                try:request=self.requests.get(timeout=.1)
                except queue.Empty:
                    now=time.monotonic()
                    if idle_timeout and now-last_request>=idle_timeout:break
                    if now-last_control>=heartbeat_interval:
                        self.pair.heartbeat();self.heartbeats+=1;last_control=time.monotonic()
                    continue
                if not request.cancelled.is_set() and time.monotonic()-request.submitted<self.queue_timeout:
                    self._execute(request)
                    last_request=last_control=time.monotonic()
                else:self._publish(request,BusyError('distributed request queue deadline exceeded'))
                request=None
        except BaseException as error:
            self.failure=error
            if request is not None:self._publish(request,RuntimeError('distributed worker session failed'))
            raise
        finally:
            self.closed.set()
            while True:
                try:waiting=self.requests.get_nowait()
                except queue.Empty:break
                self._publish(waiting,BusyError('distributed session is closed'))

    def close(self):self.closed.set()
