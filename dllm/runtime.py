"""Model residency and serialized request lifetime for the first GPU executor."""

from collections import OrderedDict
from contextlib import contextmanager
import gc
import os
import threading
import time

from .device import selected_device, synchronize, empty_cache


def require_gpu():
    if selected_device() == 'mps':
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS is not available')
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one GPU UUID")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")
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
                 host_gib=8, use_graph=True, queue_size=8, queue_timeout=120, batch_size=None):
        if (min(context, gpu_models, host_models, queue_size) <= 0
                or host_models < gpu_models or host_gib <= 0 or queue_timeout <= 0):
            raise ValueError("invalid model, residency, context, or queue limits")
        require_gpu()
        from .management import quantization_mode
        self.default_quantization = quantization_mode()
        self.models = dict(models)
        self.quantizations = {alias:self.default_quantization for alias in self.models}
        self.registry_lock = threading.RLock()
        self.context = context
        self.gpu_limit, self.host_limit = gpu_models, host_models
        self.host_bytes = int(host_gib * 1024**3)
        self.use_graph = use_graph
        self.batch_size = batch_size
        self.queue_timeout = queue_timeout
        self.host = OrderedDict()
        self.gpu = OrderedDict()
        self.lock = threading.Lock()
        self.admission = threading.BoundedSemaphore(queue_size)

    def model_ids(self):
        with self.registry_lock:
            return list(self.models)

    def _status(self):
        with self.registry_lock:
            models = [{'id': alias, 'quantization':self.quantizations[alias], 'residency': 'gpu' if alias in self.gpu else 'host' if alias in self.host else 'disk',
                       'host_weight_bytes': self.host[alias].bytes if alias in self.host else None}
                      for alias in self.models]
        return {'models': models, 'device': selected_device(), 'context': self.context,
                'host_cached_bytes': sum(model.bytes for model in self.host.values()),
                'host_budget_bytes': self.host_bytes, 'gpu_model_limit': self.gpu_limit,
                'host_model_limit': self.host_limit}

    def manage(self, action, model=None, path=None, release_host=False, quantization=None):
        """Called by the GPU worker between request groups, or under this lock."""
        from .management import command
        command(action, model, path, release_host, quantization)
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise BusyError('model management timeout')
        try:
            if action == 'register':
                import json
                from pathlib import Path
                from tokenizers import Tokenizer
                from .models import config as read_config, weight_bytes
                candidate = Path(path).expanduser().resolve()
                policy = self.default_quantization if quantization is None else quantization
                with self.registry_lock:
                    existing = self.models.get(model)
                    if existing is not None:
                        if Path(existing).expanduser().resolve() != candidate or self.quantizations[model] != policy:
                            raise ValueError('alias already refers to another snapshot or quantization')
                        return self._status()
                try:
                    config = read_config(candidate)
                    Tokenizer.from_file(str(candidate/'tokenizer.json'))
                    json.loads((candidate/'tokenizer_config.json').read_text())
                    estimated = weight_bytes(candidate,quantization=policy)
                except Exception as error:
                    raise ValueError('invalid local model snapshot') from error
                if ((config.get('task') != 'embedding' and self.context > config['max_position_embeddings'])
                        or estimated > self.host_bytes):
                    raise ValueError('model exceeds configured context or host capacity')
                with self.registry_lock:
                    self.models[model] = candidate
                    self.quantizations[model] = policy
            elif action != 'status':
                if model not in self.models:
                    raise ValueError('unknown model')
                if action == 'load':
                    executor, tier, seconds = self._activate(model)
                    return {**self._status(), 'source_tier': tier, 'activation_seconds': seconds}
                if model in self.gpu:
                    self._evict(model)
                if release_host:
                    self.host.pop(model, None)
                    gc.collect()
                    empty_cache()
            return self._status()
        finally:
            self.lock.release()

    def _evict(self, alias):
        import torch
        synchronize()
        del self.gpu[alias]
        gc.collect()
        empty_cache()

    def _activate(self, alias):
        import torch
        from .models import load, weight_bytes
        start = time.perf_counter()
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
            if model.bytes > self.host_bytes:
                raise MemoryError("model exceeds configured host weight budget")
            while sum(m.bytes for m in self.host.values()) + model.bytes > self.host_bytes:
                victim = next(iter(self.host))
                if victim in self.gpu:
                    self._evict(victim)
                del self.host[victim]
            self.host[alias] = model
        self.host.move_to_end(alias)
        while len(self.gpu) >= self.gpu_limit:
            self._evict(next(iter(self.gpu)))
        # Failure never publishes a partially initialized executor as ready.
        try:
            executor = make_executor(self.host[alias], self.use_graph, self.batch_size)
        except Exception:
            gc.collect()
            empty_cache()
            raise
        self.gpu[alias] = executor
        synchronize()
        return executor, tier, time.perf_counter() - start

    @contextmanager
    def session(self, alias):
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

    def generate(self, alias, messages, max_tokens=128, ignore_eos=False, enable_thinking=None,
                 temperature=0, top_p=1, top_k=0, seed=None, stop=None, raw_prompt=None, response_format=None,
                 presence_penalty=0, frequency_penalty=0, repetition_penalty=1, logit_bias=None,
                 logprobs=False,top_logprobs=0):
        """Yield token IDs/text deltas plus a final usage/timing event.

        Closing this generator releases its model lease, including on client
        disconnect. A single lease spans loading, prefill, and all decoding.
        """
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        from .sampling import Sampling, Sampler
        from .stopping import stop_sequences, text_delta
        stops = stop_sequences(stop)
        sampler = Sampler(Sampling(temperature, top_p, top_k, seed, presence_penalty, frequency_penalty, repetition_penalty, logit_bias))
        from .logprobs import options as score_options,decoder_for
        sampler.logprobs_top=score_options(logprobs,top_logprobs)
        if stops and sampler.logprobs_top is not None:
            raise ValueError('logprobs with custom text stops is not yet supported')
        from .structured import prepare_options, Constraint
        schema, enable_thinking = prepare_options(response_format, stops, enable_thinking, ignore_eos)
        start = time.perf_counter()
        with self.session(alias) as (executor, tier, load_seconds):
            model = executor.model
            if sampler.logprobs_top is not None:
                sampler.token_decoder=decoder_for(model)
            from .prompts import encode
            if schema is not None and raw_prompt is None and model.config["model_type"] in ("qwen3", "qwen3_5_text"):
                enable_thinking = False
            prompt = encode(model, messages, raw_prompt, enable_thinking)
            if len(prompt) + max_tokens > self.context:
                raise ValueError("prompt plus max_tokens exceeds configured context")
            if schema is not None:
                sampler.constraint = Constraint(model, schema)
            if sampler.has_adjustments:
                sampler.prepare(prompt,model.config['vocab_size'])
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
                    decoded = model.tokenizer.decode(output, skip_special_tokens=True)
                    delta, emitted, text_stop = text_delta(decoded, emitted, stops, stop or index == max_tokens-1)
                    stop = stop or text_stop
                    event={"token": token, "text": delta}
                    if sampler.logprobs_top is not None and token not in model.eos:
                        event['logprobs']=sampler.token_decoder.format(sampler.last_logprobs)
                    yield event
                    if stop:
                        finish = "stop"
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
                              "weight_quantization": model.quantization,
                              "residual_dtype": str(executor.residual_dtype).removeprefix("torch."),
                              "activation_seconds": load_seconds,
                              "first_token_seconds": first_time, "total_seconds": elapsed,
                              "decode_tokens_per_second": (len(output) - 1) / (elapsed - first_time)
                              if len(output) > 1 and elapsed > first_time else None}}


def make_executor(model, use_graph=True, max_batch=None, **kwargs):
    """Choose an execution path from the validated architecture, not the model name."""
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
    if model.config.get('task') == 'embedding':
        from .encoder import EncoderExecutor
        return EncoderExecutor(model,use_graph=use_graph)
    if model.config['model_type'] == 'gpt2':
        from .gpt2 import Gpt2Executor
        if kwargs.pop('fuse_projections',True) is False:
            raise ValueError('GPT-2 requires packed checkpoint projections')
        return Gpt2Executor(model,use_graph,max_batch or 1,**kwargs)
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
        return HybridExecutor(model, use_graph, max_batch or 1, **kwargs)
    if max_batch is not None:
        from .batch import BatchExecutor
        return BatchExecutor(model, use_graph, max_batch, **kwargs)
    from .qwen2 import Executor
    return Executor(model, use_graph, **kwargs)
