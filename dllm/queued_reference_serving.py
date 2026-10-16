"""Optional single-owner queue for explicit CPU reference HTTP generation."""
from contextlib import closing
from dataclasses import dataclass,field
import json
import queue
import threading
import time

from .distributed_admission import DistributedAdmission
from .error_state import detached_error
from .reference_serving import ReferenceRuntime
from .request_scope import cache_scope
from .runtime import BusyError
from .reference_cancellation import cancellation_scope

BUFFER_EVENTS=8


@dataclass
class ReferenceRequest:
    messages: list
    maximum: int
    options: dict
    submitted: float
    events: queue.Queue=field(default_factory=lambda:queue.Queue(BUFFER_EVENTS+1))
    cancelled: threading.Event=field(default_factory=threading.Event)
    started: threading.Event=field(default_factory=threading.Event)
    cache_scope: object=field(default_factory=cache_scope.get,repr=False)


class QueuedReferenceRuntime(ReferenceRuntime):
    """One generation/cache owner; HTTP threads only enqueue and consume events."""
    def __init__(self,runner,alias='model',*,queue_size=8,queue_timeout=120):
        if type(queue_size) is not int or not 1<=queue_size<=64:raise ValueError('CPU queue size must be 1..64')
        if type(queue_timeout) not in (int,float) or not 0<queue_timeout<=300:raise ValueError('CPU queue timeout must be within 300 seconds')
        super().__init__(runner,alias)
        self.requests=DistributedAdmission(queue_size,queue_timeout);self.queue_timeout=queue_timeout
        self.admission_lock=threading.RLock();self.accepted={};self.draining=False;self.failure=None
        self.worker=threading.Thread(target=self._work,name='dllm-cpu-reference')
        self.worker.start()

    def _admit(self):
        if self.stopping.is_set():raise BusyError('CPU reference queue is closing')
        if self.draining:raise BusyError('CPU reference queue is draining')

    def scheduler_status(self):
        with self.admission_lock:
            pending=list(self.accepted.values());queued=[r for r in pending if not r.started.is_set()]
            return dict(closed=self.closed,closing=self.stopping.is_set(),failed=self.failure is not None,
                draining=self.draining,drained=self.draining and not pending and not self.stopping.is_set(),
                worker_busy=any(r.started.is_set() for r in pending),queue_capacity=self.requests.maxsize,
                queue_timeout_seconds=self.queue_timeout,batch_capacity=1,client_admission_turns=True,
                queued_generation_requests=len(queued),worker_owned_generation_requests=len(pending)-len(queued),
                queued_commands=0,worker_owned_commands=0,stream_buffer_events=BUFFER_EVENTS,
                oldest_queued_seconds=max((max(0,time.monotonic()-r.submitted) for r in queued),default=0))

    def drain(self):
        with self.admission_lock:self.draining=True;return self.scheduler_status()

    def manage(self,action,**options):
        result=super().manage(action,**options);state=self.scheduler_status()
        return result|dict(queue_enabled=True,queued=state['queued_generation_requests'],
            queue_capacity=self.requests.maxsize,draining=self.draining,failed=self.failure is not None,
            stream_buffer_events=BUFFER_EVENTS,client_admission_turns=True)

    def generate(self,alias,messages,max_tokens=128,**options):
        if alias!=self.alias:raise ValueError('unknown CPU reference model')
        if type(max_tokens) is not int or not 1<=max_tokens<=4096:raise ValueError('CPU reference output limit must be 1..4096')
        with self.admission_lock:self._admit()
        # The bounded snapshot cannot be changed by a caller after admission.
        try:encoded=json.dumps([messages,options],ensure_ascii=False,allow_nan=False).encode('utf-8')
        except (TypeError,ValueError,RecursionError) as error:raise ValueError('CPU queued requests must be JSON values') from error
        if len(encoded)>1024**2:raise ValueError('CPU queued request exceeds 1 MiB')
        messages,options=json.loads(encoded)
        request=ReferenceRequest(messages,max_tokens,options,time.monotonic())
        with self.admission_lock:
            self._admit()
            try:self.requests.put_nowait(request)
            except queue.Full:raise BusyError('CPU reference request queue is full') from None
            self.accepted[id(request)]=request
        try:
            while True:
                try:event=request.events.get(timeout=.05)
                except queue.Empty:
                    if self.stopping.is_set():raise BusyError('CPU reference queue is closing')
                    if not request.started.is_set() and time.monotonic()-request.submitted>=self.queue_timeout:
                        raise BusyError('CPU reference request queue deadline exceeded')
                    continue
                if isinstance(event,BaseException):raise event
                yield event
                if 'finish_reason' in event:return
        finally:
            request.cancelled.set()
            with self.admission_lock:
                if self.requests.discard_cancelled(request):
                    self.accepted.pop(id(request),None)

    def _publish(self,request,event,*,terminal=False):
        # Reserve a ninth entry for a final/error; at most eight data events wait.
        while not terminal and request.events.qsize()>=BUFFER_EVENTS:
            if request.cancelled.wait(.01) or self.stopping.is_set():return False
        if request.cancelled.is_set() or (self.stopping.is_set() and not terminal):return False
        if isinstance(event,BaseException):event=detached_error(event)
        request.events.put_nowait(event);return True

    def _retire(self,request):
        with self.admission_lock:self.accepted.pop(id(request),None)

    def _flush(self):
        with self.admission_lock:
            while True:
                try:request=self.requests.get_nowait()
                except queue.Empty:break
                self._publish(request,BusyError('CPU reference queue is closing'),terminal=True)
                self.accepted.pop(id(request),None)

    def _generate_owned(self,request):
        return ReferenceRuntime.generate(self,self.alias,request.messages,request.maximum,**request.options)

    def _work(self):
        try:
            while not self.stopping.is_set():
                try:request=self.requests.get(timeout=.05)
                except queue.Empty:continue
                try:
                    if request.cancelled.is_set():continue
                    if time.monotonic()-request.submitted>=self.queue_timeout:
                        self._publish(request,BusyError('CPU reference request queue deadline exceeded'),terminal=True)
                        continue
                    request.started.set();started=time.monotonic();final=None
                    with cancellation_scope(request.cancelled.is_set),closing(self._generate_owned(request)) as generation:
                        for event in generation:
                            if request.cancelled.is_set() or self.stopping.is_set():break
                            if not isinstance(event,dict) or final is not None:
                                raise RuntimeError('invalid queued generation event ordering')
                            if 'finish_reason' in event:
                                if (not isinstance(event['finish_reason'],str) or not event['finish_reason']
                                        or not isinstance(event.get('usage'),dict)
                                        or not isinstance(event.get('timing'),dict)):
                                    raise RuntimeError('invalid queued generation terminal event')
                                final=dict(event)
                            elif not self._publish(request,event):break
                    if not request.cancelled.is_set() and not self.stopping.is_set():
                        if final is None:raise RuntimeError('queued generation ended without a terminal event')
                        if self.cleanup_pending():raise RuntimeError('queued generation cleanup remains pending')
                        timing=dict(final['timing']);timing['queue_seconds']=started-request.submitted
                        timing['total_seconds']=time.monotonic()-request.submitted
                        self._publish(request,dict(final,timing=timing),terminal=True)
                except Exception as error:
                    self._publish(request,error,terminal=True)
                    if self.cleanup_pending():
                        self.failure=detached_error(error);self.stopping.set()
                finally:
                    if self.cleanup_pending() and self.failure is None:
                        self.failure=RuntimeError('CPU reference cleanup remains pending');self.stopping.set()
                    self._retire(request)
        except BaseException as error:
            self.failure=detached_error(error);self.stopping.set()
        finally:self._flush()

    def close(self):
        with self.admission_lock:self.stopping.set()
        self._flush()
        if threading.current_thread() is self.worker:raise BusyError('CPU queue owner cannot join itself')
        self.worker.join(timeout=5)
        if self.worker.is_alive():raise BusyError('CPU queue shutdown is waiting for active execution')
        super().close()
