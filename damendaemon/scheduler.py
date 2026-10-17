"""Continuous decode batching with bounded admission and cancellable streams.

One worker owns the GPU. Requests join/free KV slots between decode steps.
Model groups drain before switching; a queued different model prevents new
admission to the current group, avoiding starvation and active-model eviction.
"""
from collections import deque
import copy
import math
import traceback
from dataclasses import dataclass, field
import queue
import threading
import time

from .runtime import Runtime, BusyError
from .error_state import detached_error
from .sampling import Sampling, Sampler, sample_rows
from .stopping import stop_sequences, text_delta
from .prompts import decode_output
from .admission import wait_for_batch, validate_queue_limits
from .request_scope import cache_scope
from .fair_admission import FairAdmission
from .client_decode_turns import ClientDecodeTurns
from .client_prefill_turns import ClientPrefillTurns
from .token_observation import TokenObservation


class PagedCleanupError(ExceptionGroup):
    """Reservations remain owned; this executor cannot safely serve more work."""


@dataclass(eq=False)
class Pending:
    alias: str
    messages: list
    max_tokens: int
    enable_thinking: bool | None = None
    sampler: Sampler = field(default_factory=lambda: Sampler(Sampling()))
    stops: tuple = ()
    raw_prompt: str | list | None = None
    schema: str | None = None
    tools: list | None = None
    tool_json: bool = False
    command: dict | None = None
    start: float = field(default_factory=time.perf_counter)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8))
    done: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    final: dict | None = None
    output: list = field(default_factory=list)
    emitted: str = ""
    first_time: float | None = None
    token_observation: TokenObservation = field(default_factory=TokenObservation)
    last_progress: float = field(default_factory=time.monotonic)
    max_batch_seen: int = 1
    slot: int = -1
    tier: str = "gpu"
    activation: float = 0
    queue_seconds: float | None = None
    cached_tokens: int = 0
    speculative_prompt: tuple = ()
    speculative_blocks: int = 0
    speculative_proposed: int = 0
    speculative_matched: int = 0
    speculative_misses: int = 0
    speculative_cooldown: int = 0
    reasoning_effort: str | None = None
    output_decoder: object | None = None
    cache_scope: object = field(default_factory=cache_scope.get, repr=False)


class BatchedRuntime:
    admitted_count = 0
    released_count = 0

    def __init__(self, models, batch_size=4, queue_size=8, queue_timeout=120, prefill_chunk_size=0, prefix_cache_tokens=0, batch_wait_ms=0, client_decode_turns=False, client_prefill_turns=False, **kwargs):
        if type(client_decode_turns) is not bool:raise ValueError("client decode turns must be boolean")
        if type(client_prefill_turns) is not bool:raise ValueError('client prefill turns must be boolean')
        if client_prefill_turns and (type(prefill_chunk_size) is not int or not 1<=prefill_chunk_size<=2**20):
            raise ValueError('client prefill turns require explicit bounded prompt chunks')
        self.client_prefill_turns=client_prefill_turns;self.prefill_turns=ClientPrefillTurns()
        self.client_decode_turns=client_decode_turns
        self.decode_turns=ClientDecodeTurns()
        validate_queue_limits(queue_size, queue_timeout)
        if type(batch_size) is not int or not 1 <= batch_size <= 16 or queue_size < batch_size:
            raise ValueError("batch size must be 1..16 and fit inside the request queue")
        if type(prefill_chunk_size) is not int or prefill_chunk_size<0:
            raise ValueError('prefill chunk size must be a nonnegative integer')
        if type(prefix_cache_tokens) is not int or prefix_cache_tokens<0:
            raise ValueError('prefix cache tokens must be a nonnegative integer')
        if type(batch_wait_ms) not in (int,float) or not math.isfinite(batch_wait_ms) or not 0<=batch_wait_ms<=10:
            raise ValueError("batch wait must be a finite number between 0 and 10 milliseconds")
        self.batch_wait_seconds=batch_wait_ms/1000
        self.prefix_cache_tokens=prefix_cache_tokens
        self.prefill_chunk_size=prefill_chunk_size
        from .partition_policy import selected_split
        from .hybrid_pipeline_policy import budgets
        if budgets() is not None and prefix_cache_tokens:
            raise ValueError('hybrid pipeline does not support prefix snapshots')
        if selected_split() is not None and (prefill_chunk_size or prefix_cache_tokens):
            raise ValueError('layer partition does not yet support chunked prefill or prefix snapshots')
        self.runtime = Runtime(models, batch_size=batch_size, queue_size=queue_size, queue_timeout=queue_timeout, **kwargs)
        self.context = kwargs.get('context', 1024)
        if prefill_chunk_size or prefix_cache_tokens:
            from .device import selected_device
            if selected_device()=='cuda':
                # Load offset-attention dispatch before the server advertises
                # readiness, not on the first partial-prefix hit.
                from torch.nn.attention.bias import causal_lower_right
        self.models = self.runtime.models
        self.batch_size = batch_size
        self.queue_timeout = queue_timeout
        self.queue_capacity = queue_size
        self.admission = threading.BoundedSemaphore(queue_size)
        self.admitted_count = 0
        self.released_count = 0
        self.condition = threading.Condition()
        self.pending = deque()
        self.fair_admission = FairAdmission()
        self.requests = set()
        self.closed = False
        self.draining = False
        self.worker_busy = False
        self.worker = threading.Thread(target=self._work, name="damendaemon-gpu", daemon=True)
        self.worker.start()

    def model_ids(self):
        return self.runtime.model_ids()

    def is_healthy(self):
        with self.condition:
            return not self.closed and not self.draining and self.worker.is_alive()

    def scheduler_status(self):
        """Read queue ownership without GPU access or waiting for the worker."""
        with self.condition:
            queued = set(self.pending)
            live = self.requests - {request for request in self.requests if request.done.is_set()}
            queued &= live
            owned = live - queued
            now = time.perf_counter()
            return {
                "closed": self.closed,
                "draining": self.draining,
                "drained": self.draining and not live and not self.worker_busy,
                "worker_busy": self.worker_busy,
                "queue_capacity": self.queue_capacity,
                "queue_timeout_seconds": self.queue_timeout,
                "batch_capacity": self.batch_size,
                "client_decode_turns": self.client_decode_turns,
                "client_prefill_turns": self.client_prefill_turns,
                "queued_generation_requests": sum(r.command is None for r in queued),
                "worker_owned_generation_requests": sum(r.command is None for r in owned),
                "queued_commands": sum(r.command is not None for r in queued),
                "worker_owned_commands": sum(r.command is not None for r in owned),
                "oldest_queued_seconds": max((max(0, now-r.start) for r in queued), default=0),
                "admission_outstanding": self.admitted_count-self.released_count,
                "admitted_total": self.admitted_count,
                "released_total": self.released_count,
                "admission_semaphore": getattr(self.admission, '_value', None),
            }

    def drain(self):
        """Stop new admission; the worker finishes previously admitted work."""
        with self.condition:
            self.draining = True
            self.condition.notify_all()
            return self.scheduler_status()

    def model_manifest(self):
        # Registry-only snapshot; never wait for the GPU worker or admission.
        return self.runtime.model_manifest()

    def manage(self, action, model=None, path=None, release_host=False, quantization=None, chat_template=None):
        from .management import command
        return self._submit(command(action, model, path, release_host, quantization, chat_template))

    def embed(self, alias, inputs, encoding_format="float", dimensions=None):
        from .embeddings import options
        inputs, encoding_format, dimensions = options(inputs, encoding_format, dimensions,max_length=self.context)
        if not isinstance(alias, str) or alias not in self.models:
            raise ValueError("unknown model")
        return self._submit({"action":"embed", "model":alias, "inputs":inputs,
                             "encoding_format":encoding_format, "dimensions":dimensions})

    def rerank(self,alias,query,documents,top_n=None,return_documents=False):
        from .reranking import options
        query,documents,top_n,return_documents=options(query,documents,top_n,return_documents)
        if not isinstance(alias,str) or alias not in self.models:raise ValueError('unknown model')
        return self._submit({'action':'rerank','model':alias,'query':query,'documents':documents,
            'top_n':top_n,'return_documents':return_documents})

    def score(self,alias,prompt,candidate_token_ids):
        from .scoring import options
        prompt,candidates=options(prompt,candidate_token_ids,max_length=self.context)
        if not isinstance(alias,str) or alias not in self.models:raise ValueError('unknown model')
        return self._submit({'action':'score','model':alias,'prompt':prompt,'candidate_token_ids':candidates})

    def _submit(self, options):
        request = Pending(options.get('model') or '', None, 0, command=options)
        if not self.admission.acquire(blocking=False):
            raise BusyError('request queue is full')
        with self.condition:
            self.admitted_count += 1
        try:
            with self.condition:
                if self.closed or self.draining:
                    raise BusyError('runtime is closed' if self.closed else 'runtime is draining')
                self.requests.add(request)
                self.pending.append(request)
                self.condition.notify_all()
            # A busy GPU worker cannot expire its own queue until the current
            # group yields. Bound the caller's wait, removing only work that
            # the worker has not taken ownership of. In-flight mutations must
            # still report their actual outcome instead of executing after a
            # caller has been told they were cancelled.
            timeout = min(self.queue_timeout, 1.0) if options['action'] == 'status' else self.queue_timeout
            if not request.done.wait(timeout):
                with self.condition:
                    if request in self.pending:
                        self.pending.remove(request)
                        request.cancelled.set()
                        self._finish(request, BusyError('model management queue timeout'))
                        self.condition.notify_all()
                request.done.wait()
            if request.error:
                raise request.error
            return request.final
        finally:
            request.cancelled.set()
            with self.condition:
                try:
                    self.pending.remove(request)
                except ValueError:
                    pass
                self.requests.discard(request)
                self.released_count += 1
                self.condition.notify_all()
            self.admission.release()

    def generate(self, alias, messages, max_tokens=128, enable_thinking=None,
                 temperature=0, top_p=1, top_k=0, seed=None, stop=None, raw_prompt=None, response_format=None,
                 presence_penalty=0, frequency_penalty=0, repetition_penalty=1, logit_bias=None,
                 logprobs=False,top_logprobs=0,tools=None,tool_choice=None,reasoning_effort=None):
        if alias not in self.models:
            raise ValueError("unknown model")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_tokens > self.context:
            raise ValueError('max_tokens exceeds configured context')
        sampler = Sampler(Sampling(temperature, top_p, top_k, seed, presence_penalty, frequency_penalty, repetition_penalty, logit_bias))
        stops = stop_sequences(stop)
        from .tool_calls import prepare_tools
        tools,tool_schema=prepare_tools(tools,tool_choice)
        if tools and (raw_prompt is not None or response_format is not None or logprobs or stops):
            raise ValueError('tool calls cannot combine with raw prompts, structured output, scores or custom stops')
        from .logprobs import options as score_options
        sampler.logprobs_top=score_options(logprobs,top_logprobs)
        if stops and sampler.logprobs_top is not None:
            raise ValueError('logprobs with custom text stops is not yet supported')
        if raw_prompt is not None:
            from .prompts import validate_raw
            raw_prompt = validate_raw(raw_prompt,max_length=self.context)
            if messages is not None or enable_thinking is not None or reasoning_effort is not None:
                raise ValueError('raw prompts cannot include chat messages or template options')
        from .structured import prepare_options
        schema, enable_thinking = prepare_options(response_format, stops, enable_thinking,
                                                  reasoning_effort=reasoning_effort)
        if tool_schema is not None:
            if enable_thinking is True:
                raise ValueError('forced tool calls cannot use thinking or ignore_eos')
            schema=tool_schema
        # Copying/constructing the request can fail before it enters the queue.
        # Acquire capacity only once cleanup can own a fully prepared request.
        request = Pending(alias, copy.deepcopy(messages), max_tokens, enable_thinking, sampler, stops, raw_prompt, schema,tools=tools,tool_json=tool_schema is not None,reasoning_effort=reasoning_effort)
        if not self.admission.acquire(blocking=False):
            raise BusyError("request queue is full")
        with self.condition:
            self.admitted_count += 1
        try:
            with self.condition:
                if self.closed or self.draining:
                    raise BusyError("runtime is closed" if self.closed else "runtime is draining")
                self.requests.add(request)
                self.pending.append(request)
                self.condition.notify_all()
            while True:
                request.ready.clear()
                try:
                    event = request.events.get_nowait()
                except queue.Empty:
                    if request.done.is_set():
                        # A producer can enqueue its last block and finish
                        # between get_nowait() observing empty and this check.
                        if not request.events.empty():continue
                        if request.error:
                            raise request.error
                        if request.final:
                            yield request.final
                        return
                    # Only queued work has a queue deadline. Check ownership
                    # under the same lock as _next so a timeout cannot cancel
                    # a request already taken by the GPU worker.
                    timeout = None
                    with self.condition:
                        if request in self.pending:
                            timeout = request.start + self.queue_timeout - time.perf_counter()
                            if timeout <= 0:
                                self.pending.remove(request)
                                self._finish(request, BusyError("request queue timeout"))
                                self.condition.notify_all()
                                continue
                    request.ready.wait(timeout)
                    continue
                with self.condition:
                    self.condition.notify_all()
                yield event
        finally:
            request.cancelled.set()
            with self.condition:
                try:
                    self.pending.remove(request)
                except ValueError:
                    pass
                self.requests.discard(request)
                self.released_count += 1
                self.condition.notify_all()
            self.admission.release()

    def _finish(self, request, error=None):
        with self.condition:
            if request.done.is_set():
                return
            if error is not None:
                # Do not transport worker frames (and their GPU tensors) to the
                # caller. A retained exception must not pin an evicted model.
                detached = detached_error(error)
                if error.__traceback__ is not None:
                    detached.add_note('Worker traceback:\n'+''.join(traceback.format_tb(error.__traceback__)))
                error = detached
            request.error = error
            request.done.set()
            request.ready.set()
            self.requests.discard(request)

    def _emit(self, request, token, executor, prompt_count, *, defer_finish=False):
        # Paged callers release their KV reservation before marking done.
        if request.cancelled.is_set():
            if not defer_finish:self._finish(request)
            return True
        if request.sampler.constraint is not None:
            request.sampler.constraint.accept(token)
        request.sampler.observe(token)
        request.output.append(token)
        now = time.perf_counter()
        request.token_observation.observe(now - request.start, sequential=not request.speculative_blocks)
        if request.first_time is None:
            request.first_time = now - request.start
        request.last_progress = time.monotonic()
        stop = token in executor.model.eos
        finished = stop or len(request.output) >= request.max_tokens
        reasoning=''
        if request.output_decoder is not None:
            delta,reasoning,text_stop=request.output_decoder.feed(token,final=finished)
        else:
            decoded = decode_output(executor.model, request.output)
            delta, request.emitted, text_stop = text_delta(decoded, request.emitted, request.stops, finished)
        stop = stop or text_stop
        finished = finished or text_stop
        event={"token": token, "text": delta}
        if request.output_decoder is not None:
            event['tool_call_format']='harmony'
            if request.output_decoder.calls:event['tool_calls']=request.output_decoder.calls
        if reasoning:event['reasoning_content']=reasoning
        if request.sampler.logprobs_top is not None and token not in executor.model.eos:
            event['logprobs']=request.sampler.token_decoder.format(request.sampler.last_logprobs)
        request.events.put_nowait(event)
        request.ready.set()
        if finished:
            elapsed = time.perf_counter() - request.start
            count = len(request.output)
            reason="tool_calls" if request.output_decoder is not None and request.output_decoder.calls else "stop" if stop else "length"
            request.final = {"finish_reason": reason, "usage": {
                "prompt_tokens": prompt_count, "completion_tokens": count,
                "total_tokens": prompt_count + count}, "timing": {
                "source_tier": request.tier, "activation_seconds": request.activation,
                "queue_seconds": request.queue_seconds,
                "token_observation": request.token_observation.report(),
                "cached_prompt_tokens":request.cached_tokens,
                "speculative_blocks":request.speculative_blocks,
                "speculative_proposed_drafts":request.speculative_proposed,
                "speculative_matched_drafts":request.speculative_matched,
                "kernel_backend": executor.kernel_backend,
                "weight_quantization": getattr(executor.model, "weight_format", getattr(executor.model, "quantization", "none")), "max_batch_size": request.max_batch_seen,
                "residual_dtype": str(executor.residual_dtype).removeprefix("torch."),
                "first_token_seconds": request.first_time, "total_seconds": elapsed,
                "decode_tokens_per_second": (count-1)/(elapsed-request.first_time)
                if count > 1 and elapsed > request.first_time else None}}
            if not defer_finish:self._finish(request)
        return finished

    def _next(self, alias=None):
        """Caller holds the condition; client turns never cross group barriers."""
        while self.pending:
            first=self.pending[0]
            now=time.perf_counter()
            if first.cancelled.is_set() or now-first.start>self.queue_timeout:
                self.pending.popleft()
                self._finish(first,None if first.cancelled.is_set() else BusyError('request queue timeout'))
                continue
            if alias is not None and (first.command is not None or first.alias!=alias):
                return None
            if first.command is not None:
                self.fair_admission.reset()
                first.queue_seconds=max(0,now-first.start)
                return self.pending.popleft()
            # Inspect only the leading same-model generation group. Compact its
            # expired/cancelled entries in one pass, avoiding repeated deque
            # removal and preserving every later command/model barrier.
            live=[];scanned=0
            for request in self.pending:
                if request.command is not None or request.alias!=first.alias:break
                scanned+=1
                if request.cancelled.is_set():self._finish(request)
                elif now-request.start>self.queue_timeout:self._finish(request,BusyError('request queue timeout'))
                else:live.append(request)
            for _ in range(scanned):self.pending.popleft()
            if not live:continue # cancellation can arrive without taking this condition
            chosen=self.fair_admission.choose(first.alias,live)
            self.pending.extendleft(reversed([request for request in live if request is not chosen]))
            chosen.queue_seconds=max(0,now-chosen.start)
            return chosen
        self.fair_admission.reset()
        return None

    def _ready_decode(self,active):
        ready=[(slot,request,length) for slot,(request,length) in active.items() if not request.events.full()]
        return self.decode_turns.choose(ready) if self.client_decode_turns else ready

    def _prepare(self,request,executor):
        from .prompts import encode,validate_lengths,decoder_prefix
        thinking=request.enable_thinking
        if request.schema is not None and request.raw_prompt is None and executor.model.config['model_type'] in ('qwen3','qwen3_5_text'):
            thinking=False
        tokens=encode(executor.model,request.messages,request.raw_prompt,thinking,request.tools,request.tool_json,request.reasoning_effort)
        validate_lengths(executor,len(tokens),request.max_tokens)
        from .harmony import output_stream
        request.output_decoder=output_stream(executor.model,request.raw_prompt,request.stops,
            request.schema,request.sampler.logprobs_top,tools=request.tools)
        if request.schema is not None:
            from .structured import Constraint
            request.sampler.constraint=Constraint(executor.model,request.schema)
        if request.sampler.has_adjustments:
            request.sampler.prepare(decoder_prefix(executor,tokens),executor.model.config['vocab_size'])
        if request.sampler.logprobs_top is not None:
            from .logprobs import decoder_for
            request.sampler.token_decoder=decoder_for(executor.model)
        if getattr(executor,'speculative_verifier',None) is not None:
            from .speculative import eligible
            if eligible(request) and not self.prefill_chunk_size and not self.prefix_cache_tokens:
                request.speculative_prompt=tuple(tokens[-1024:])
                # Room for two verification blocks overlaps GPU work with the
                # consumer draining the previous block. Keep the same queue
                # object because the consumer may already be waiting on it.
                with request.events.mutex:
                    request.events.maxsize=max(8,2*executor.speculative_verifier.width)
        return tokens

    def _speculate(self,ready,active,executor):
        verifier=getattr(executor,'speculative_verifier',None)
        if verifier is None or len(active)!=1 or len(ready)!=1:return False
        slot,request,prompt_count=ready[0]
        if not request.speculative_prompt:return False
        if request.speculative_cooldown:
            request.speculative_cooldown-=1
            return False
        remaining=request.max_tokens-len(request.output)
        start=executor.lengths[slot]
        capacity=request.events.maxsize-request.events.qsize()
        limit=min(verifier.width-1,remaining-1,capacity-1)
        if limit<1 or start+verifier.width>executor.context:
            return False
        from .speculative import propose
        history=(list(request.speculative_prompt)+request.output[-1024:])[-1024:]
        drafts=propose(history,limit)
        if not drafts:return False
        tokens,matched=verifier.verify(request.output[-1],drafts,start,slot)
        request.speculative_blocks+=1;request.speculative_proposed+=len(drafts)
        request.speculative_matched+=matched
        request.speculative_misses=0 if matched else request.speculative_misses+1
        if request.speculative_misses>=2:
            request.speculative_cooldown=8;request.speculative_misses=0
        previous=len(request.output)
        try:
            for token in tokens:
                if self._emit(request,token,executor,prompt_count):
                    del active[slot]
                    break
        finally:
            # The last emitted token remains the next decode input. Rejected,
            # filler and post-EOS positions cannot extend the committed prefix.
            executor.lengths[slot]=start+len(request.output)-previous
        return True

    def _prefill_slot(self,partial,chunk_size):
        if not self.client_prefill_turns:return next(iter(partial))
        return self.prefill_turns.choose([(slot,request,min(chunk_size,len(tokens)-start))
            for slot,(request,tokens,start) in partial.items()])[0]

    def _chunked_group(self,first,executor,tier,activation):
        first.tier,first.activation=tier,activation
        active,partial={},{}
        cache=getattr(executor,'prefix_cache',None)
        if self.prefix_cache_tokens and cache is None:
            from .prefix import PrefixCache
            cache=executor.prefix_cache=PrefixCache(executor,self.prefix_cache_tokens)
        chunk_size=self.prefill_chunk_size or executor.context
        next_request=first
        try:
            while next_request is not None or active or partial:
                if next_request is not None:
                    incoming=[next_request]
                    next_request=None
                    with self.condition:
                        while len(incoming)+len(active)+len(partial)<self.batch_size:
                            request=self._next(first.alias)
                            if request is None:break
                            incoming.append(request)
                    for request in incoming:
                        try:
                            if request.cancelled.is_set():
                                self._finish(request)
                                continue
                            tokens=self._prepare(request,executor)
                            request.slot=next(i for i in range(self.batch_size) if i not in active and i not in partial)
                            reused,logits=cache.restore(tokens,request.slot,scope=request.cache_scope) if cache is not None else (0,None)
                            request.cached_tokens=reused
                            if logits is not None:
                                token=request.sampler.select(logits.squeeze(0)).item()
                                if not self._emit(request,token,executor,len(tokens)):
                                    active[request.slot]=(request,len(tokens))
                            else:
                                partial[request.slot]=(request,tokens,reused)
                        except Exception as error:
                            self._finish(request,error)
                for slot,(request,_,__) in list(partial.items()):
                    if request.cancelled.is_set():
                        self._finish(request)
                        del partial[slot]
                if not partial:self.prefill_turns.reset()
                # At most one bounded prompt chunk before each decode step.
                if partial:
                    slot=self._prefill_slot(partial,chunk_size)
                    request,tokens,start=partial[slot]
                    end=min(start+chunk_size,len(tokens))
                    try:
                        logits=executor.prefill_chunk(tokens[start:end],slot,start,final=end==len(tokens))
                        if end==len(tokens):
                            if cache is not None:cache.remember(tokens,slot,logits,scope=request.cache_scope)
                            token=request.sampler.select(logits.squeeze(0)).item()
                            if not self._emit(request,token,executor,len(tokens)):
                                active[slot]=(request,len(tokens))
                            del partial[slot]
                        else:
                            # Rotate unfinished prompts so each admitted slot
                            # gets a turn before this prompt's next chunk.
                            del partial[slot]
                            partial[slot]=(request,tokens,end)
                    except Exception as error:
                        self._finish(request,error)
                        del partial[slot]
                if not partial:self.prefill_turns.reset()
                for slot,(request,_) in list(active.items()):
                    if request.cancelled.is_set():
                        self._finish(request)
                        del active[slot]
                    elif request.events.full() and time.monotonic()-request.last_progress>self.queue_timeout:
                        self._finish(request,BusyError('stream backpressure timeout'))
                        del active[slot]
                ready=self._ready_decode(active)
                if ready:
                    logits,next_tokens=executor.decode_batch([r.output[-1] for _,r,_ in ready],[slot for slot,_,_ in ready])
                    selected=sample_rows(logits,[r.sampler for _,r,_ in ready],next_tokens)
                    for (slot,request,length),token in zip(ready,selected):
                        request.max_batch_seen=max(request.max_batch_seen,len(ready))
                        if self._emit(request,token,executor,length):del active[slot]
                with self.condition:
                    if len(active)+len(partial)<self.batch_size:
                        next_request=self._next(first.alias)
                    if active and not ready and not partial and next_request is None:
                        self.condition.wait(timeout=.02)
        except Exception as error:
            for request,_ in active.values():self._finish(request,error)
            for request,_,__ in partial.values():self._finish(request,error)
            if next_request:self._finish(next_request,error)
            raise

    def _paged_group(self, first, executor, tier, activation):
        from .pages import PageCapacityError
        first.tier, first.activation = tier, activation
        active, partial, waiting = {}, {}, deque([(first, None)])
        cache=getattr(executor,'prefix_cache',None)
        if self.prefix_cache_tokens and cache is None:
            from .prefix import PrefixCache
            cache=executor.prefix_cache=PrefixCache(executor,self.prefix_cache_tokens)
        chunk_size = ((self.prefill_chunk_size or executor.context)
                      if (self.prefill_chunk_size or cache is not None) and getattr(executor,'supports_chunked_prefill',False) else 0)
        owned = set()
        prepared = []
        try:
            while waiting or active or partial:
                with self.condition:
                    while len(waiting)+len(active)+len(partial)<self.batch_size:
                        request=self._next(first.alias)
                        if request is None:
                            break
                        waiting.append((request,None))
                for request,tokens in list(waiting):
                    if request.cancelled.is_set() or time.monotonic()-request.last_progress>self.queue_timeout:
                        self._finish(request, None if request.cancelled.is_set() else BusyError('KV admission timeout'))
                        waiting.remove((request,tokens))
                prepared=[]
                while waiting and len(active)+len(partial)+len(prepared)<self.batch_size:
                    request,tokens=waiting[0]
                    slot=next(i for i in range(self.batch_size) if i not in owned)
                    try:
                        if tokens is None:
                            tokens=self._prepare(request,executor)
                            waiting[0]=(request,tokens)
                        executor.reserve_request(slot,len(tokens)+request.max_tokens)
                    except PageCapacityError:
                        break  # Preserve FIFO while existing reservations finish.
                    except Exception as error:
                        self._finish(request,error)
                        waiting.popleft()
                        continue
                    request.slot=slot
                    owned.add(slot)
                    waiting.popleft()
                    prepared.append((request,tokens))
                if chunk_size:
                    for request,tokens in prepared:
                        try:
                            reused,logits=cache.restore(tokens,request.slot,scope=request.cache_scope) if cache is not None else (0,None)
                            request.cached_tokens=reused
                            if logits is None:
                                partial[request.slot]=(request,tokens,reused)
                            else:
                                token=request.sampler.select(logits.squeeze(0)).item()
                                if self._emit(request,token,executor,len(tokens),defer_finish=True):
                                    executor.release_request(request.slot)
                                    owned.remove(request.slot)
                                    self._finish(request)
                                else:
                                    active[request.slot]=(request,len(tokens))
                        except Exception as error:
                            self._finish(request,error)
                            executor.release_request(request.slot)
                            owned.remove(request.slot)
                    prepared=[]
                if prepared:
                    try:
                        logits=executor.prefill_batch([tokens for _,tokens in prepared],[r.slot for r,_ in prepared])
                        selected=sample_rows(logits,[r.sampler for r,_ in prepared])
                        for (request,tokens),token in zip(prepared,selected):
                            if self._emit(request,token,executor,len(tokens),defer_finish=True):
                                executor.release_request(request.slot)
                                owned.remove(request.slot)
                                self._finish(request)
                            else:
                                active[request.slot]=(request,len(tokens))
                    except Exception as error:
                        for request,_ in prepared:
                            self._finish(request,error)
                            active.pop(request.slot,None)
                            if request.slot in owned:
                                executor.release_request(request.slot)
                                owned.remove(request.slot)
                    prepared=[]
                for slot,(request,_,__) in list(partial.items()):
                    if request.cancelled.is_set():
                        self._finish(request)
                        executor.release_request(slot)
                        owned.remove(slot)
                        del partial[slot]
                if not partial:self.prefill_turns.reset()
                # Bound prompt work between decode steps, keeping its reservation.
                if partial:
                    slot=self._prefill_slot(partial,chunk_size)
                    request,tokens,start=partial[slot]
                    end=min(start+chunk_size,len(tokens))
                    try:
                        logits=executor.prefill_chunk(tokens[start:end],slot,start,final=end==len(tokens))
                        if end==len(tokens):
                            if cache is not None:
                                cache.remember(tokens,slot,logits,scope=request.cache_scope)
                            token=request.sampler.select(logits.squeeze(0)).item()
                            if self._emit(request,token,executor,len(tokens),defer_finish=True):
                                executor.release_request(slot)
                                owned.remove(slot)
                                self._finish(request)
                            else:
                                active[slot]=(request,len(tokens))
                            del partial[slot]
                        else:
                            # Rotate unfinished prompts so each admitted slot
                            # gets a turn before this prompt's next chunk.
                            del partial[slot]
                            partial[slot]=(request,tokens,end)
                    except Exception as error:
                        self._finish(request,error)
                        executor.release_request(slot)
                        owned.remove(slot)
                        del partial[slot]
                if not partial:self.prefill_turns.reset()
                for slot,(request,_) in list(active.items()):
                    timed_out=request.events.full() and time.monotonic()-request.last_progress>self.queue_timeout
                    if request.cancelled.is_set() or timed_out:
                        self._finish(request,BusyError('stream backpressure timeout') if timed_out else None)
                        executor.release_request(slot)
                        owned.remove(slot)
                        del active[slot]
                ready=self._ready_decode(active)
                if ready:
                    logits,next_ids=executor.decode_batch([r.output[-1] for _,r,_ in ready],[slot for slot,_,_ in ready])
                    selected=sample_rows(logits,[r.sampler for _,r,_ in ready],next_ids)
                    for (slot,request,length),token in zip(ready,selected):
                        request.max_batch_seen=max(request.max_batch_seen,len(ready))
                        if self._emit(request,token,executor,length,defer_finish=True):
                            executor.release_request(slot)
                            owned.remove(slot)
                            self._finish(request)
                            del active[slot]
                with self.condition:
                    if not waiting and len(active)+len(partial)<self.batch_size:
                        request=self._next(first.alias)
                        if request is not None:
                            waiting.append((request,None))
                    if (active or waiting) and not ready and not prepared and not partial:
                        self.condition.wait(timeout=.02)
        except Exception as error:
            for request,_ in [*active.values(),*waiting,*prepared]:
                self._finish(request,error)
            for request,_,__ in partial.values():
                self._finish(request,error)
            raise
        finally:
            cleanup_errors = []
            for slot in owned:
                try:
                    executor.release_request(slot)
                except Exception as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise PagedCleanupError('paged reservation cleanup failed', cleanup_errors)

    def _group(self, first, executor, tier, activation):
        if self.client_prefill_turns and not getattr(executor,'supports_chunked_prefill',False):
            raise ValueError('client prefill turns require a chunk-capable executor')
        if self.prefix_cache_tokens and not getattr(executor, 'supports_prefix_cache', True):
            raise ValueError('this architecture does not yet support prefix snapshots')
        if getattr(executor,'requires_full_prefill',False) and (self.prefill_chunk_size or self.prefix_cache_tokens):
            raise ValueError('this architecture does not yet support prompt chunking or prefix snapshots')
        if hasattr(executor,'reserve_request'):
            return self._paged_group(first,executor,tier,activation)
        if (self.prefill_chunk_size or self.prefix_cache_tokens) and getattr(executor,'supports_chunked_prefill',False):
            return self._chunked_group(first,executor,tier,activation)
        first.tier, first.activation = tier, activation
        active = {}  # slot -> (request, prompt length)
        next_request = first
        try:
            while next_request is not None or active:
                if next_request is not None:
                    incoming = [next_request]
                    next_request = None
                    with self.condition:
                        while len(incoming) + len(active) < self.batch_size:
                            request = self._next(first.alias)
                            if request is None:
                                break
                            incoming.append(request)
                    prepared = []
                    free_slots = iter(i for i in range(self.batch_size) if i not in active)
                    for request in incoming:
                        try:
                            if request.cancelled.is_set():
                                self._finish(request)
                                continue
                            tokens = self._prepare(request,executor)
                            request.slot = next(free_slots)
                            prepared.append((request, tokens))
                        except Exception as error:
                            self._finish(request, error)
                    if prepared:
                        try:
                            logits = executor.prefill_batch(
                                [tokens for _, tokens in prepared],
                                [request.slot for request, _ in prepared])
                            first_tokens = sample_rows(logits, [r.sampler for r, _ in prepared])
                            for (request, tokens), token in zip(prepared, first_tokens):
                                if not self._emit(request, token, executor, len(tokens)):
                                    active[request.slot] = (request, len(tokens))
                        except Exception as error:
                            for request, _ in prepared:
                                self._finish(request, error)
                                active.pop(request.slot, None)
                for slot, (request, _) in list(active.items()):
                    if request.cancelled.is_set():
                        self._finish(request)
                        del active[slot]
                    elif request.events.full() and time.monotonic() - request.last_progress > self.queue_timeout:
                        self._finish(request, BusyError("stream backpressure timeout"))
                        del active[slot]
                ready = self._ready_decode(active)
                if ready and not self._speculate(ready,active,executor):
                    logits, next_tokens = executor.decode_batch([r.output[-1] for _, r, _ in ready],
                                                           [slot for slot, _, _ in ready])
                    selected = sample_rows(logits, [r.sampler for _, r, _ in ready], next_tokens)
                    for (slot, request, length), token in zip(ready, selected):
                        request.max_batch_seen = max(request.max_batch_seen, len(ready))
                        if self._emit(request, token, executor, length):
                            del active[slot]
                with self.condition:
                    if len(active) < self.batch_size:
                        next_request = self._next(first.alias)
                    if active and not ready and next_request is None:
                        self.condition.wait(timeout=0.02)
        except Exception as error:
            for request, _ in active.values():
                self._finish(request, error)
            if next_request:
                self._finish(next_request, error)
            raise

    def _work(self):
        try:
            self._run_work()
        except BaseException as error:
            # A dead owner cannot service either taken or queued requests.
            # Keep device cleanup on the external close path, and do not retain
            # the fatal exception or its frames in request state.
            failure = RuntimeError(f'scheduler worker stopped unexpectedly ({type(error).__name__})')
            with self.condition:
                self.closed = True
                for request in tuple(self.requests):
                    request.cancelled.set()
                    self._finish(request, failure)
                self.pending.clear()
                self.worker_busy = False
                self.condition.notify_all()

    def _run_work(self):
        while True:
            with self.condition:
                first = self._next()
                while first is None and not self.closed:
                    self.condition.wait()
                    first = self._next()
                if self.closed:
                    return
                self.worker_busy = True
            if first.command is not None:
                try:
                    if first.command['action'] == 'embed':
                        options = first.command
                        first.final = self.runtime.embed(options['model'], options['inputs'],
                                                         options['encoding_format'], options['dimensions'])
                    elif first.command['action']=='rerank':
                        options=first.command
                        first.final=self.runtime.rerank(options['model'],options['query'],options['documents'],
                            options['top_n'],options['return_documents'])
                    elif first.command['action']=='score':
                        options=first.command
                        first.final=self.runtime.score(options['model'],options['prompt'],options['candidate_token_ids'])
                    else:
                        first.final = self.runtime.manage(**first.command)
                    self._finish(first)
                except Exception as error:
                    self._finish(first, error)
                finally:
                    with self.condition:
                        self.worker_busy = False
                        self.condition.notify_all()
                continue
            try:
                with self.runtime.session(first.alias) as (executor, tier, activation):
                    wait_for_batch(self,first,self.batch_wait_seconds)
                    self.decode_turns.reset();self.prefill_turns.reset()
                    self._group(first, executor, tier, activation)
            except PagedCleanupError:
                # The group already tried every reservation. Let the fatal-owner
                # path close admission and wake queued callers; close() owns
                # final runtime cleanup outside this worker.
                raise
            except Exception as error:
                if not first.done.is_set():
                    self._finish(first, error)
            finally:
                # A worker local otherwise retains the previous model while the
                # next session tries to evict it and reserve replacement VRAM.
                executor = None
                self.decode_turns.reset();self.prefill_turns.reset()
                with self.condition:
                    self.worker_busy = False
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            for request in tuple(self.requests):
                # A command removed from pending is owned by the worker and
                # may already be mutating residency. Let it report its actual
                # outcome; close still waits for the worker before cleanup.
                if request.command is not None and request not in self.pending:
                    continue
                request.cancelled.set()
                self._finish(request, BusyError("runtime closed"))
            self.condition.notify_all()
        self.worker.join(timeout=30)
        if self.worker.is_alive():
            raise BusyError('scheduler shutdown is waiting for its active worker')
        if hasattr(self.runtime, 'close'):
            self.runtime.close()
