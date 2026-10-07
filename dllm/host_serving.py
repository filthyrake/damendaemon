"""Explicit CPU host-worker adapter for the existing distributed HTTP owner."""
from types import SimpleNamespace
from .cpu_reference_text import CpuReferenceText
from .distributed_serving import DistributedRuntime,STREAM_BUFFER_EVENTS
from .host_hybrid_worker import HostHybridWorker


class HostServingPair:
    """Transfer a ready coordinator and matching prepared tokenizer to HTTP.

    Caller runs the authenticated peer's step loop. Only the distributed owner
    submits inference commands; close may interrupt it from another thread.
    """
    rank=0
    packed_prefill=False

    def __init__(self,worker,text,*,chunk_size=128):
        if not isinstance(worker,HostHybridWorker) or not isinstance(text,CpuReferenceText):
            raise ValueError('host serving needs a prepared worker and CPU tokenizer')
        with worker.lock:
            if (worker.rank!=0 or not worker.ready or worker.failed or worker.closed
                    or any(worker.channel.state.lengths) or getattr(worker,'_completion_owner',None) is not None
                    or worker.mapped_vocab_size!=text.vocab_size or not text.excluded_ids<=worker.excluded_ids
                    or bool(text.eos_ids&worker.excluded_ids)
                    or type(chunk_size) is not int or not 1<=chunk_size<=4096
                    or max(worker.stage.max_batch,min(chunk_size,worker.stage.context))*worker.width*4>worker.channel.transport.max_frame_bytes):
                raise ValueError('incompatible, occupied or claimed CPU host serving session')
            self.worker,self.text,self.chunk_size=worker,text,chunk_size
            self.max_batch=worker.stage.max_batch
            self.state=worker.channel.state
            self.identity=worker.channel.transport.identity
            self.model=SimpleNamespace(max_context=worker.stage.context,quantization=worker.stage.model.quantization,
                eos=text.eos_ids,tokenizer=text.tokenizer,raw_tokens=self._tokens,chat_tokens=self._chat)
            worker._completion_owner=object()

    def _tokens(self,prompt):
        tokens=self.text.tokens(prompt,max_context=self.model.max_context,max_new_tokens=1,chunk_size=self.chunk_size)
        self.worker._tokens(tokens)
        return tokens

    def _chat(self,*args,**kwargs):raise ValueError('CPU host serving accepts raw completions only')

    def prefill(self,tokens,slot):
        for start in range(0,len(tokens),self.chunk_size):
            result=self.worker.prefill_chunk(tokens[start:start+self.chunk_size],slot,start)
        return result

    def decode(self,tokens,slots):return self.worker.decode(tokens,slots)

    def step(self,tokens):
        # Compatibility with the single-slot distributed owner protocol.
        if not self.state.lengths[0]:return self.prefill(tokens,0)
        if len(tokens)!=1:raise ValueError('single-slot decode requires one token')
        return self.decode(tokens,[0])[0]

    def release(self,slots=None):return self.worker.release([0] if slots is None else slots)
    def heartbeat(self):return self.worker.ping()
    def stop(self):return self.worker.stop()
    def close(self):self.worker.close()


class HostDistributedRuntime(DistributedRuntime):
    """Raw HTTP serving over CPU peers; all performance metadata stays explicit."""
    source_tier='cpu_reference'
    kernel_backend='cpu-host-reference'

    def __init__(self,pair,alias='model',queue_size=8,queue_timeout=120):
        if not isinstance(pair,HostServingPair):raise ValueError('CPU host runtime requires its explicit pair adapter')
        super().__init__(pair,alias,queue_size,queue_timeout)

    def model_manifest(self):raise ValueError('CPU host sessions cannot be exported as GPU model manifests')

    def manage(self,action,**options):
        if action!='status' or options:raise ValueError('CPU host session model management is unsupported')
        stage=self.pair.worker.stage
        return dict(device='cpu',scope='cpu_host_reference',official_model_qualified=False,
            context=self.pair.model.max_context,
            models=[dict(id=self.alias,residency='cpu',quantization=self.pair.model.quantization,
                         session_identity=self.pair.identity.hex())],
            distributed=True,stage_layers=[[0,stage.layer_end],[stage.layer_end,stage.c['num_hidden_layers']]],
            max_batch_size=self.max_batch,packed_prefill=False,active=self.active,queued=self.requests.qsize(),
            completed=self.completed,heartbeats=self.heartbeats,
            stream_buffer_events=STREAM_BUFFER_EVENTS,
            failed=self.failure is not None,closed=self.closed.is_set())

    def generate(self,alias,messages,max_tokens,**options):
        if type(max_tokens) is not int or not 1<=max_tokens<=4096:raise ValueError('CPU host output limit must be 1..4096')
        yield from super().generate(alias,messages,max_tokens,**options)

    def run(self,*args,**kwargs):
        try:return super().run(*args,**kwargs)
        except BaseException as primary:
            # A failed epoch is never returned to an owner for request reuse.
            try:self.pair.close()
            except BaseException:primary.add_note('CPU host cleanup failed; retain pair and retry close')
            raise

    def close(self):
        super().close()
        self.pair.close() # transport close interrupts blocked peer I/O


def handler_for_host(runtime,auth=None):
    """Reuse ordinary HTTP validation/auth while reporting failed peer health."""
    if not isinstance(runtime,HostDistributedRuntime):raise ValueError('host HTTP handler requires CPU host runtime')
    from .server import handler_for
    base=handler_for(runtime,auth)
    class Handler(base):
        def do_GET(self):
            if self.path!='/health':return super().do_GET()
            if not self.authorize():return
            healthy=not runtime.closed.is_set() and runtime.failure is None and runtime.pair.worker.ready
            self.send_json(200 if healthy else 503,dict(status='ok' if healthy else 'closed',
                scope='cpu_host_reference',official_model_qualified=False))
    return Handler
