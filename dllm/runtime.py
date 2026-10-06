"""Model residency and serialized request lifetime for the first GPU executor."""

from collections import OrderedDict
from contextlib import contextmanager
import gc
import os
import threading
import time

from .device import selected_device, selected_accelerator, synchronize, empty_cache
from .prompts import decode_output


def require_gpu():
    from .partition_policy import selected_split
    split=selected_split()
    from .hybrid_pipeline_policy import budgets, validate_environment
    pipeline_budgets = budgets()
    if pipeline_budgets is not None:
        validate_environment()
    if selected_device() == 'xpu':
        import torch
        from .xpu_policy import validate_runtime
        validate_runtime(torch)
        return
    if selected_device() == 'mps':
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS is not available')
        return
    if selected_accelerator() == 'rocm':
        from .rocm_policy import configure_attention
        configure_attention()
        import torch
        from .rocm_policy import validate_runtime, configure_allocator
        validate_runtime(torch)
        torch.cuda.set_device(0)
        configure_allocator(torch)
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    count=len(pipeline_budgets) if pipeline_budgets is not None else 2 if split is not None else 1
    devices=visible.split(',')
    if len(devices)!=count or len(set(devices))!=count or any(not value.startswith('GPU-') for value in devices):
        raise ValueError(f"set CUDA_VISIBLE_DEVICES to exactly {count} distinct GPU UUIDs")
    import torch
    if getattr(getattr(torch, 'version', None), 'hip', None):
        raise RuntimeError('HIP PyTorch requires explicit --device rocm selection')
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError(f"exactly {count} CUDA GPUs must be visible")
    torch.cuda.set_device(0)


@contextmanager
def request_kv(executor, tokens):
    paged = hasattr(executor, 'reserve_request')
    if paged:
        executor.reserve_request(0, tokens)
    try:
        yield
    finally:
        if paged:
            executor.release_request(0)


class BusyError(Exception):
    pass


class Runtime:
    def __init__(self, models, context=1024, gpu_models=1, host_models=2,
                 host_gib=8, use_graph=True, queue_size=8, queue_timeout=120, batch_size=None, chat_templates=None,
                 quantizations=None):
        if selected_device() == 'xpu':
            from .xpu_policy import validate_environment, validate_limits
            validate_environment()
            validate_limits(context, 1 if batch_size is None else batch_size)
        from .admission import validate_queue_limits
        validate_queue_limits(queue_size, queue_timeout)
        if (min(context, gpu_models, host_models, queue_size) <= 0
                or host_models < gpu_models or host_gib <= 0 or queue_timeout <= 0):
            raise ValueError("invalid model, residency, context, or queue limits")
        from .chat_template import validate_override
        overrides = dict(chat_templates or {})
        if overrides.keys() - models.keys():
            raise ValueError("chat template refers to an unknown model alias")
        for alias, text in overrides.items():
            from .models import config
            validate_override(text, config(models[alias]))
        from .management import quantization_mode
        policies = dict(quantizations or {})
        if policies.keys() - models.keys():
            raise ValueError('quantization refers to an unknown model alias')
        for policy in policies.values():
            if policy is None:
                raise ValueError('per-model quantization must be explicit')
            quantization_mode(policy)
        from .hybrid_pipeline_policy import budgets, validate as validate_pipeline, validate_environment
        if budgets() is not None:
            validate_environment(use_graph)
            if (type(context) is not int or not 1 <= context <= 2048
                    or (batch_size is not None and (type(batch_size) is not int or not 1 <= batch_size <= 4))):
                raise ValueError('pipeline serving requires context <=2048 and batch 1..4')
            for alias, path in models.items():
                from .models import config
                validate_pipeline(config(path), policies.get(alias, quantization_mode()), context,
                                  1 if batch_size is None else batch_size, use_graph)
        require_gpu()
        self.default_quantization = quantization_mode()
        self.models = dict(models)
        self.quantizations = {alias:policies.get(alias, self.default_quantization) for alias in self.models}
        self.chat_templates = {alias: overrides.get(alias) for alias in self.models}
        self.registry_lock = threading.RLock()
        self.context = context
        self.gpu_limit, self.host_limit = gpu_models, host_models
        self.host_bytes = int(host_gib * 1024**3)
        self.use_graph = use_graph
        self.batch_size = batch_size
        self.queue_timeout = queue_timeout
        self.host = OrderedDict()
        self.gpu = OrderedDict()
        self.failed_builds = []
        self.failed_executors = []
        self.closing = self.closed = False
        self.lock = threading.Lock()
        self.admission = threading.BoundedSemaphore(queue_size)

    def model_ids(self):
        with self.registry_lock:
            return list(self.models)

    def model_manifest(self):
        from .model_manifest import snapshot
        with self.registry_lock:
            models = dict(self.models)
            policies = dict(self.quantizations)
            templates = dict(self.chat_templates)
        return snapshot(models, policies, templates)

    def _status(self):
        import torch
        from .chat_template import override_metadata
        with self.registry_lock:
            models = [{'id': alias, 'quantization':self.quantizations[alias], 'residency': 'gpu' if alias in self.gpu else 'host' if alias in self.host else 'disk',
                       'chat_template': override_metadata(self.chat_templates[alias]),
                       'host_weight_bytes': self.host[alias].bytes if alias in self.host else None,
                       'vision_host_weight_bytes': getattr(getattr(self.host.get(alias),'vision_weights',None),'bytes',0),
                       'vision_gpu_weight_bytes': getattr(getattr(self.gpu.get(alias),'vision',None),'bytes',0),
                       'gpu_projection_cache_bytes': getattr(self.gpu.get(alias), 'packed_cache_bytes', 0),
                       'gpu_packed_layout_overhead_bytes': getattr(self.gpu.get(alias), 'packed_layout_overhead_bytes', 0),
                       'q4_marlin_abi': getattr(self.gpu.get(alias), 'q4_marlin_abi', None),
                       'partition': getattr(self.gpu.get(alias),'residency',None),
                       'partition_rebind_count': getattr(self.gpu.get(alias),'rebind_count',None),
                       'output_head_workspace_bytes': getattr(self.gpu.get(alias), 'output_head_workspace_bytes', 0),
                       'weight_streaming': (self.gpu[alias].offload.status()
                            if getattr(self.gpu.get(alias), 'offload', None) is not None else None),
                       'pipeline': (self.gpu[alias].status()
                            if hasattr(self.gpu.get(alias), 'placement') else None),
                       'executor_failed': (getattr(self.gpu.get(alias),'failed',False)
                            or getattr(self.gpu.get(alias),'_activation_failed',False))}
                      for alias in self.models]
        from .partition_policy import selected_split
        from .hybrid_pipeline_policy import budgets
        pipeline_budgets = budgets()
        devices=None
        if (selected_split() is not None or pipeline_budgets is not None) and torch.cuda.is_initialized():
            visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            devices=[dict(index=i,uuid=visible[i],name=torch.cuda.get_device_name(i),
                          allocated_bytes=torch.cuda.memory_allocated(i),reserved_bytes=torch.cuda.memory_reserved(i))
                     for i in range(len(pipeline_budgets) if pipeline_budgets is not None else 2)]
        cuda_initialized = selected_device() == 'cuda' and torch.cuda.is_initialized()
        allocated=torch.cuda.memory_allocated() if cuda_initialized else None
        xpu_initialized = selected_device() == 'xpu' and torch.xpu.is_initialized()
        if devices is not None:allocated=sum(d['allocated_bytes'] for d in devices)
        peak_allocated = torch.cuda.max_memory_allocated() if allocated is not None and devices is None else None
        return {'models': models, 'device': selected_accelerator(), 'context': self.context,
                'closing':self.closing, 'closed':self.closed,
                'cuda_allocated_bytes': allocated if selected_accelerator()=='cuda' else None,
                'cuda_peak_allocated_bytes': peak_allocated if selected_accelerator()=='cuda' else None,
                'hip_allocated_bytes': allocated if selected_accelerator()=='rocm' else None,
                'rocm_aotriton_experimental': os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL')
                    if selected_accelerator()=='rocm' else None,
                'cuda_devices':devices,
                'pending_pipeline_cleanup':len(self.failed_builds),
                'pending_executor_cleanup':len(self.failed_executors),
                'xpu_allocated_bytes': torch.xpu.memory_allocated() if xpu_initialized else None,
                'xpu_reserved_bytes': torch.xpu.memory_reserved() if xpu_initialized else None,
                'xpu_peak_allocated_bytes': torch.xpu.max_memory_allocated() if xpu_initialized else None,
                'mps_allocated_bytes': torch.mps.current_allocated_memory() if selected_device()=='mps' else None,
                'host_cached_bytes': sum(model.bytes for model in self.host.values()),
                'host_budget_bytes': self.host_bytes, 'gpu_model_limit': self.gpu_limit,
                'host_model_limit': self.host_limit}

    def manage(self, action, model=None, path=None, release_host=False, quantization=None, chat_template=None):
        """Called by the GPU worker between request groups, or under this lock."""
        from .management import command
        command(action, model, path, release_host, quantization, chat_template)
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise BusyError('model management timeout')
        try:
            if self.closing and action != 'status':
                raise BusyError('runtime is closing or closed')
            if action in ('unload', 'remove'):
                self._cleanup_failed_owners()
            if action == 'register':
                import json
                from pathlib import Path
                from tokenizers import Tokenizer
                from .models import config as read_config, weight_bytes, is_gguf
                candidate = Path(path).expanduser().resolve()
                policy = self.default_quantization if quantization is None else quantization
                with self.registry_lock:
                    existing = self.models.get(model)
                    if existing is not None:
                        if (Path(existing).expanduser().resolve() != candidate or self.quantizations[model] != policy
                                or self.chat_templates[model] != chat_template):
                            raise ValueError('alias already refers to another snapshot, quantization or chat template')
                        return self._status()
                try:
                    config = read_config(candidate)
                    from .hybrid_pipeline_policy import validate as validate_pipeline
                    validate_pipeline(config, policy, self.context, self.batch_size or 1, self.use_graph)
                    if chat_template is not None:
                        from .chat_template import validate_override
                        validate_override(chat_template, config)
                    if not is_gguf(candidate):
                        Tokenizer.from_file(str(candidate/'tokenizer.json'))
                        json.loads((candidate/'tokenizer_config.json').read_text())
                    estimated = weight_bytes(candidate,quantization=policy)
                except Exception as error:
                    raise ValueError('invalid local model snapshot') from error
                if ((config.get('task') not in ('embedding','rerank') and self.context > config['max_position_embeddings'])
                        or estimated > self.host_bytes):
                    raise ValueError('model exceeds configured context or host capacity')
                with self.registry_lock:
                    self.models[model] = candidate
                    self.quantizations[model] = policy
                    self.chat_templates[model] = chat_template
            elif action != 'status':
                if model not in self.models:
                    raise ValueError('unknown model')
                if action == 'load':
                    executor, tier, seconds = self._activate(model)
                    return {**self._status(), 'source_tier': tier, 'activation_seconds': seconds}
                if model in self.gpu:
                    self._evict(model)
                if release_host or action == 'remove':
                    self.host.pop(model, None)
                    gc.collect()
                    empty_cache()
                if action == 'remove':
                    with self.registry_lock:
                        del self.models[model]
                        del self.quantizations[model]
                        del self.chat_templates[model]
            return self._status()
        finally:
            self.lock.release()

    def _evict(self, alias):
        import torch
        synchronize()
        if hasattr(self.gpu[alias],'close'):self.gpu[alias].close()
        del self.gpu[alias]
        gc.collect()
        empty_cache()

    def _cleanup_failed_owners(self):
        # Failed construction or rebinding may still own device buffers. Never
        # start another load until each owner has completed retryable cleanup.
        for owners in (self.failed_builds, self.failed_executors):
            while owners:
                owners[0].close()
                owners.pop(0)

    def close(self):
        """Drain ownership under the lifetime lock; failed cleanup is retryable."""
        self.closing = True
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise BusyError('runtime shutdown is waiting for active requests')
        try:
            if self.closed:
                return
            self._cleanup_failed_owners()
            while self.gpu:
                self._evict(next(iter(self.gpu)))
            self.host.clear()
            gc.collect()
            self.closed = True
        finally:
            self.lock.release()

    def _activate(self, alias):
        import torch
        from .models import load, weight_bytes
        start = time.perf_counter()
        if self.closing:
            raise BusyError('runtime is closing or closed')
        self._cleanup_failed_owners()
        if alias in self.gpu and (getattr(self.gpu[alias],'failed',False)
                                or getattr(self.gpu[alias],'_activation_failed',False)):
            self._evict(alias)
        if alias in self.gpu:
            self.gpu.move_to_end(alias)
            self.host.move_to_end(alias)
            return self.gpu[alias], "gpu", time.perf_counter() - start
        tier = "host" if alias in self.host else "disk"
        if alias not in self.host:
            estimated = weight_bytes(self.models[alias],quantization=self.quantizations[alias])
            if estimated > self.host_bytes:
                raise MemoryError("model exceeds configured host weight budget")
            # Evict host entries only when their GPU executors are also idle;
            # this method always runs under the request-lifetime lock.
            while (len(self.host) >= self.host_limit or
                   sum(m.bytes for m in self.host.values()) + estimated > self.host_bytes):
                victim = next(iter(self.host))
                if victim in self.gpu:
                    self._evict(victim)
                del self.host[victim]
            model = load(self.models[alias], self.context,quantization=self.quantizations[alias])
            if self.chat_templates[alias] is not None:
                model.configure_tokenizer(model.tokenizer, self.chat_templates[alias], model.special_tokens)
            if model.bytes > self.host_bytes:
                raise MemoryError("model exceeds configured host weight budget")
            while sum(m.bytes for m in self.host.values()) + model.bytes > self.host_bytes:
                victim = next(iter(self.host))
                if victim in self.gpu:
                    self._evict(victim)
                del self.host[victim]
            self.host[alias] = model
        self.host.move_to_end(alias)
        if len(self.gpu)>=self.gpu_limit:
            victim=next(iter(self.gpu))
            resident=self.gpu[victim]
            if hasattr(resident,'can_rebind') and resident.can_rebind(self.host[alias]):
                # The lifetime lock has drained all users of the old alias.
                # Remove it before mutating graph-owned weight storage; a
                # partially copied model must never be published under either ID.
                del self.gpu[victim]
                try:
                    resident.rebind(self.host[alias])
                except BaseException as error:
                    self.failed_executors.append(resident)
                    try:
                        self._cleanup_failed_owners()
                        gc.collect()
                        empty_cache()
                    except BaseException as cleanup_error:
                        error.add_note(f'executor cleanup incomplete; retained for retry: {type(cleanup_error).__name__}: {cleanup_error}')
                    raise
                self.gpu[alias]=resident
                return resident,tier,time.perf_counter()-start
            # An incompatible candidate must not stay alive in this frame
            # while the ordinary eviction path allocates its replacement.
            resident=None
        while len(self.gpu) >= self.gpu_limit:
            self._evict(next(iter(self.gpu)))
        # Failure never publishes a partially initialized executor as ready.
        try:
            executor = make_executor(self.host[alias], self.use_graph, self.batch_size)
        except Exception as error:
            from .hybrid_pipeline_policy import budgets
            if budgets() is not None:
                from .hybrid_pipeline_build import PipelineBuildError
                if isinstance(error, PipelineBuildError) and not error.closed:
                    self.failed_builds.append(error)
            gc.collect()
            empty_cache()
            raise
        self.gpu[alias] = executor
        try:
            synchronize()
        except BaseException:
            # Retain ownership for retryable eviction, but never reuse an
            # executor whose final activation completion failed.
            executor._activation_failed = True
            raise
        return executor, tier, time.perf_counter() - start

    @contextmanager
    def session(self, alias):
        if self.closing:
            raise BusyError('runtime is closing or closed')
        if alias not in self.models:
            raise ValueError("unknown model")
        if not self.admission.acquire(blocking=False):
            raise BusyError("request queue is full")
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=self.queue_timeout)
            if not acquired:
                raise BusyError("request queue timeout")
            yield self._activate(alias)
        finally:
            if acquired:
                self.lock.release()
            self.admission.release()

    def embed(self, alias, inputs, encoding_format='float', dimensions=None):
        from .embeddings import options
        inputs, encoding_format, dimensions = options(inputs, encoding_format, dimensions)
        start = time.perf_counter()
        with self.session(alias) as (executor, tier, activation):
            model = executor.model
            if getattr(model, 'task', None) != 'embedding':
                raise ValueError('model does not support embeddings')
            width = model.config['hidden_size']
            if dimensions is not None and dimensions != width:
                raise ValueError('this encoder does not support dimension reduction')
            tokens = model.encode(inputs)
            values = executor.embed(tokens).float().cpu().tolist()
            if encoding_format == 'base64':
                import base64
                import struct
                values = [base64.b64encode(struct.pack('<'+str(width)+'f',*row)).decode('ascii') for row in values]
            count = sum(map(len,tokens))
            return {'object':'list', 'model':alias,
                    'data':[{'object':'embedding','index':i,'embedding':row} for i,row in enumerate(values)],
                    'usage':{'prompt_tokens':count,'total_tokens':count},
                    'dllm_timing':{'source_tier':tier,'activation_seconds':activation,
                        'total_seconds':time.perf_counter()-start,'kernel_backend':executor.kernel_backend,
                        'compute_dtype':str(executor.compute_dtype).removeprefix('torch.')}}

    def rerank(self,alias,query,documents,top_n=None,return_documents=False):
        from .reranking import options,ranked
        query,documents,top_n,return_documents=options(query,documents,top_n,return_documents)
        start=time.perf_counter()
        with self.session(alias) as (executor,tier,activation):
            if getattr(executor.model,'task',None)!='rerank':raise ValueError('model does not support reranking')
            tokens,types=executor.model.encode_pairs(query,documents)
            scores=executor.rerank(tokens,types).float().cpu().tolist()
            count=sum(map(len,tokens))
            return {'model':alias,'results':ranked(scores,documents,top_n,return_documents),
                'score_type':'raw_relevance','calibrated':False,
                'usage':{'prompt_tokens':count,'total_tokens':count},
                'dllm_timing':{'source_tier':tier,'activation_seconds':activation,
                    'total_seconds':time.perf_counter()-start,'kernel_backend':executor.kernel_backend,
                    'compute_dtype':str(executor.compute_dtype).removeprefix('torch.')}}

    def score(self,alias,prompt,candidate_token_ids):
        import torch
        from .scoring import options
        prompt,candidates=options(prompt,candidate_token_ids)
        start=time.perf_counter()
        with self.session(alias) as (executor,tier,activation):
            if not hasattr(executor,'score_candidates'):
                raise ValueError('model does not support direct candidate scoring')
            tokens=executor.model.raw_tokens(prompt)
            if not tokens or len(tokens)>executor.context:
                raise ValueError('prompt is empty or exceeds context')
            with request_kv(executor,len(tokens)):
                logits=executor.score_candidates(tokens,candidates).reshape(-1)
                if not torch.isfinite(logits).all().item():raise RuntimeError('non-finite candidate scores')
                probabilities=logits.softmax(-1).cpu().tolist()
                values=logits.cpu().tolist()
            selected=max(range(len(values)),key=values.__getitem__)
            return {'object':'token_scores','model':alias,'selected_index':selected,
                'selected_token_id':candidates[selected],
                'normalization':'candidate_set','calibrated':False,
                'data':[{'index':i,'token_id':token,'logit':value,'probability':probability}
                    for i,(token,value,probability) in enumerate(zip(candidates,values,probabilities))],
                'usage':{'prompt_tokens':len(tokens),'completion_tokens':0,'total_tokens':len(tokens)},
                'dllm_timing':{'source_tier':tier,'activation_seconds':activation,
                    'total_seconds':time.perf_counter()-start,'kernel_backend':executor.kernel_backend,
                    'compute_dtype':str(executor.compute_dtype).removeprefix('torch.')}}

    def generate(self, alias, messages, max_tokens=128, ignore_eos=False, enable_thinking=None,
                 temperature=0, top_p=1, top_k=0, seed=None, stop=None, raw_prompt=None, response_format=None,
                 presence_penalty=0, frequency_penalty=0, repetition_penalty=1, logit_bias=None,
                 logprobs=False,top_logprobs=0,tools=None,tool_choice=None,reasoning_effort=None):
        """Yield token IDs/text deltas plus a final usage/timing event.

        Closing this generator releases its model lease, including on client
        disconnect. A single lease spans loading, prefill, and all decoding.
        """
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        from .sampling import Sampling, Sampler
        from .stopping import stop_sequences, text_delta
        stops = stop_sequences(stop)
        from .tool_calls import prepare_tools
        tools,tool_schema=prepare_tools(tools,tool_choice)
        if tools and (raw_prompt is not None or response_format is not None or logprobs or stops):
            raise ValueError('tool calls cannot combine with raw prompts, structured output, scores or custom stops')
        sampler = Sampler(Sampling(temperature, top_p, top_k, seed, presence_penalty, frequency_penalty, repetition_penalty, logit_bias))
        from .logprobs import options as score_options,decoder_for
        sampler.logprobs_top=score_options(logprobs,top_logprobs)
        if stops and sampler.logprobs_top is not None:
            raise ValueError('logprobs with custom text stops is not yet supported')
        from .structured import prepare_options, Constraint
        schema, enable_thinking = prepare_options(response_format, stops, enable_thinking, ignore_eos)
        if tool_schema is not None:
            if enable_thinking is True or ignore_eos:
                raise ValueError('forced tool calls cannot use thinking or ignore_eos')
            schema=tool_schema
        start = time.perf_counter()
        with self.session(alias) as (executor, tier, load_seconds):
            model = executor.model
            from .harmony import output_stream
            output_decoder=output_stream(model,raw_prompt,stops,schema,sampler.logprobs_top,ignore_eos,tools)
            if sampler.logprobs_top is not None:
                sampler.token_decoder=decoder_for(model)
            from .prompts import encode,validate_lengths,decoder_prefix
            if schema is not None and raw_prompt is None and model.config["model_type"] in ("qwen3", "qwen3_5_text"):
                enable_thinking = False
            prompt = encode(model, messages, raw_prompt, enable_thinking,tools,tool_schema is not None,reasoning_effort)
            validate_lengths(executor,len(prompt),max_tokens)
            if schema is not None:
                sampler.constraint = Constraint(model, schema)
            if sampler.has_adjustments:
                sampler.prepare(decoder_prefix(executor,prompt),model.config['vocab_size'])
            with request_kv(executor, len(prompt)+max_tokens):
                first = sampler.select(executor.prefill(prompt).squeeze(0)).item()
                output = []
                emitted = ""
                first_time = None
                token = first
                finish = "length"
                for index in range(max_tokens):
                    if sampler.constraint is not None:
                        sampler.constraint.accept(token)
                    sampler.observe(token)
                    output.append(token)
                    if first_time is None:
                        first_time = time.perf_counter() - start
                    stop = not ignore_eos and token in model.eos
                    reasoning=''
                    if output_decoder is not None:
                        delta,reasoning,text_stop=output_decoder.feed(token,final=stop or index==max_tokens-1)
                    else:
                        decoded = decode_output(model, output)
                        delta, emitted, text_stop = text_delta(decoded, emitted, stops, stop or index == max_tokens-1)
                    stop = stop or text_stop
                    event={"token": token, "text": delta}
                    if output_decoder is not None:
                        event['tool_call_format']='harmony'
                        if output_decoder.calls:event['tool_calls']=output_decoder.calls
                    if reasoning:event['reasoning_content']=reasoning
                    if sampler.logprobs_top is not None and token not in model.eos:
                        event['logprobs']=sampler.token_decoder.format(sampler.last_logprobs)
                    yield event
                    if stop:
                        finish = "tool_calls" if output_decoder is not None and output_decoder.calls else "stop"
                        break
                    if index + 1 < max_tokens:
                        logits, next_token = executor.decode(token)
                        token = (sampler.select(logits.squeeze(0)).item()
                                 if temperature or sampler.constraint is not None or sampler.has_adjustments or sampler.logprobs_top is not None else next_token.item())
                elapsed = time.perf_counter() - start
                yield {"finish_reason": finish, "usage": {
                    "prompt_tokens": len(prompt), "completion_tokens": len(output),
                    "total_tokens": len(prompt) + len(output),
                }, "timing": {"source_tier": tier, "kernel_backend": executor.kernel_backend,
                              "weight_quantization": getattr(model,"weight_format",model.quantization),
                              "residual_dtype": str(executor.residual_dtype).removeprefix("torch."),
                              "activation_seconds": load_seconds,
                              "first_token_seconds": first_time, "total_seconds": elapsed,
                              "decode_tokens_per_second": (len(output) - 1) / (elapsed - first_time)
                              if len(output) > 1 and elapsed > first_time else None}}


def make_executor(model, use_graph=True, max_batch=None, **kwargs):
    """Choose an execution path from the validated architecture, not the model name."""
    from .partition_policy import selected_split
    from .hybrid_pipeline_policy import budgets, validate as validate_pipeline
    limits = budgets()
    if limits is not None:
        validate_pipeline(model.config, model.quantization, model.max_context,
                          max_batch or 1, use_graph)
        if kwargs:
            raise ValueError('pipeline serving does not accept executor overrides')
        from .hybrid_pipeline_build import build_cuda_pipeline
        return build_cuda_pipeline(model, list(range(len(limits))), limits, max_batch or 1)
    split=selected_split()
    if split is not None:
        if kwargs:raise ValueError('layer partition does not accept executor overrides')
        from .partition import PartitionExecutor
        return PartitionExecutor(model,split,use_graph,max_batch or 1)
    budget = os.environ.get('DLLM_KV_CACHE_TOKENS')
    if budget is not None:
        if (not budget.isascii() or not budget.isdecimal() or int(budget)<=0
                or selected_device()!='cuda' or os.environ.get('DLLM_KERNELS')=='torch'
                or model.config['model_type'] not in ('qwen2','qwen3','llama','mistral')):
            raise ValueError('paged KV requires a positive token budget and a supported native CUDA dense decoder')
        from .paged import PagedExecutor
        if kwargs.pop('fuse_projections',True) is False:
            raise ValueError('paged KV requires fused projections')
        return PagedExecutor(model,use_graph,max_batch or 1,cache_tokens=int(budget),**kwargs)
    if model.config['model_type']=='mamba2':
        from .mamba2 import Mamba2Executor
        return Mamba2Executor(model,use_graph=use_graph,max_batch=max_batch or 1,**kwargs)
    if model.config['model_type']=='mamba':
        from .mamba import MambaExecutor
        return MambaExecutor(model,use_graph,max_batch or 1,**kwargs)
    if model.config['model_type']=='t5':
        from .t5 import T5Executor
        return T5Executor(model,use_graph,max_batch or 1,**kwargs)
    if model.config['model_type'] == 'bloom':
        from .bloom import BloomExecutor
        if kwargs.pop('fuse_projections', True) is False:
            raise ValueError('BLOOM requires packed checkpoint projections')
        return BloomExecutor(model, use_graph, max_batch or 1, **kwargs)
    if model.config['model_type'] == 'gpt_neox':
        from .neox import NeoxExecutor
        if kwargs.pop('fuse_projections',True) is False:
            raise ValueError('GPT-NeoX requires packed checkpoint projections')
        return NeoxExecutor(model,use_graph,max_batch or 1,**kwargs)
    if model.config.get('task') in ('embedding','rerank'):
        from .encoder import EncoderExecutor
        return EncoderExecutor(model,use_graph=use_graph)
    if model.config['model_type'] == 'gpt2':
        from .gpt2 import Gpt2Executor
        if kwargs.pop('fuse_projections',True) is False:
            raise ValueError('GPT-2 requires packed checkpoint projections')
        return Gpt2Executor(model,use_graph,max_batch or 1,**kwargs)
    if model.config['model_type'] == 'gpt_oss':
        from .gpt_oss import GptOssExecutor
        if kwargs.pop('fuse_projections',True) is False:
            raise ValueError('GPT-OSS requires packed projections')
        return GptOssExecutor(model,use_graph,max_batch or 1,**kwargs)
    if model.config['model_type'] == 'olmoe':
        from .moe import OlmoeExecutor
        if kwargs.pop('fuse_projections', True) is False:
            raise ValueError('OLMoE requires fused projections')
        return OlmoeExecutor(model, use_graph, max_batch or 1, **kwargs)
    if model.config['model_type'] == 'starcoder2':
        from .starcoder import StarcoderExecutor
        if kwargs.pop('fuse_projections', True) is False:
            raise ValueError('StarCoder2 requires fused projections')
        return StarcoderExecutor(model, use_graph, max_batch or 1, **kwargs)
    if model.config['model_type'] == 'gemma3_text':
        from .gemma import GemmaExecutor
        if kwargs.pop('fuse_projections', True) is False:
            raise ValueError('Gemma requires fused projections')
        return GemmaExecutor(model, use_graph, max_batch or 1, **kwargs)
    if model.config['model_type'] == 'qwen3_5_text':
        from .hybrid import HybridExecutor
        if kwargs.pop('fuse_projections', True) is False:
            raise ValueError('unfused projection mode is not available for hybrid execution')
        if getattr(model,'vision_weights',None) is not None:
            from .vision_policy import validate_execution
            validate_execution(model,max_batch or 1)
            from .vision_hybrid import VisionHybridExecutor
            return VisionHybridExecutor(model,use_graph=use_graph,max_batch=max_batch or 1,**kwargs)
        return HybridExecutor(model, use_graph, max_batch or 1, **kwargs)
    if max_batch is not None:
        from .batch import BatchExecutor
        return BatchExecutor(model, use_graph, max_batch, **kwargs)
    from .qwen2 import Executor
    return Executor(model, use_graph, **kwargs)
