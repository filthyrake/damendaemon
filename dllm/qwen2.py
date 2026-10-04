"""Independent dense decoder using PyTorch/CUDA and dllm kernels, not HF generation.

Dense Qwen2, Qwen3, Llama and Phi-3; model-specific mixed precision.
The single-sequence base is extended by BatchExecutor. CUDA graphs remove
per-operation Python dispatch within a captured decode step; the host still
stages metadata, replays the graph, and reads the selected token.
"""

import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from .device import selected_device, empty_cache, available_memory, attention as scaled_attention
from safetensors import safe_open
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .config import decoder_config, eos_ids
from .rotary import inverse_frequencies, attention_factor


def weight_files(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard escapes model directory")
    return [path / name for name in names]


def weight_bytes(path,quantization=None):
    """Estimate host weight storage from headers before loading tensors."""
    from .quantization import mode
    quantization=mode(quantization)
    quantized = quantization != 'none'
    config = decoder_config(json.loads((Path(path) / 'config.json').read_text()))
    if quantization=='q4_0' and config['model_type'] not in ('qwen2','qwen3','llama','phi3','mistral'):
        raise ValueError('Q4_0 is currently supported only for dense rotary decoders')
    if config['model_type'] in ('gpt2','gpt_neox') and quantized:
        raise ValueError('GPT-2/GPT-NeoX quantization is not implemented')
    element_bytes = 4 if config['model_type'] in ('starcoder2','gpt2','gpt_neox') else 2
    total, embedding, has_head = 0, None, False
    for file in weight_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if config['model_type']=='gpt2' and key.endswith(('attn.bias','attn.masked_bias')):
                    continue
                if config['model_type']=='gpt_neox' and key.endswith(('attention.bias','attention.masked_bias','rotary_emb.inv_freq')):
                    continue
                shape = reader.get_slice(key).get_shape()
                is_embedding = key.endswith('embed_tokens.weight')
                if is_embedding:
                    embedding = shape
                has_head = has_head or key == 'lm_head.weight'
                if quantized and len(shape) == 2 and not is_embedding and shape[1] % 32 == 0:
                    total += math.prod(shape)//(2 if quantization=='q4_0' else 1) + 2 * shape[0] * (shape[1] // 32)
                else:
                    total += element_bytes * math.prod(shape)
    if quantized and embedding is not None and not has_head:
        if config.get('tie_word_embeddings'):
            total += math.prod(embedding)//(2 if quantization=='q4_0' else 1) + 2 * embedding[0] * (embedding[1] // 32)
    return total



class ModelWeights:
    def __init__(self, path, max_context, quantization=None):
        from .quantization import mode
        self.quantization = mode(quantization)
        if self.quantization != 'none' and selected_device() != 'cuda':
            raise ValueError('quantized execution currently requires CUDA')
        self.path = Path(path).resolve()
        self.config = decoder_config(json.loads((self.path / "config.json").read_text()))
        c = self.config
        if self.quantization=='q4_0' and c['model_type'] not in ('qwen2','qwen3','llama','phi3','mistral'):
            raise ValueError('Q4_0 is currently supported only for dense rotary decoders')
        if c['model_type'] in ('gpt2','gpt_neox') and self.quantization!='none':
            raise ValueError('GPT-2/GPT-NeoX quantization is not implemented')
        if not 1 <= max_context <= c["max_position_embeddings"]:
            raise ValueError("context exceeds model position limit")
        self.max_context = max_context
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        tokenizer_config = json.loads((self.path / "tokenizer_config.json").read_text())
        template = tokenizer_config.get("chat_template")
        if template is None and (self.path / "chat_template.jinja").exists():
            template = (self.path / "chat_template.jinja").read_text()
        if template is not None and not isinstance(template, str):
            raise ValueError("model needs one text chat template")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                     extensions=['jinja2.ext.loopcontrols'])
        def raise_exception(message):
            raise ValueError(message)
        environment.globals.update(raise_exception=raise_exception,
                                   strftime_now=lambda fmt: datetime.now().strftime(fmt))
        self.template = environment.from_string(template) if template is not None else None
        self.special_tokens = {key: tokenizer_config.get(key) for key in ("bos_token", "eos_token")}
        self.eos = eos_ids(c.get('eos_token_id'))
        generation = self.path / "generation_config.json"
        if generation.exists():
            self.eos.update(eos_ids(json.loads(generation.read_text()).get('eos_token_id')))
        if not self.eos:
            raise ValueError('model needs at least one EOS token ID')
        state = {}
        for candidate in weight_files(self.path):
            with safe_open(str(candidate), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if not key.startswith(('model.visual.', 'mtp.')):
                        state[key] = reader.get_tensor(key)
        if c["model_type"] in ("qwen3_5_text", "gemma3_text", "starcoder2", "olmoe", "gpt2", "gpt_neox"):
            if c['model_type'] == 'gpt_neox':
                from .neox import load_weights
            elif c['model_type'] == 'gpt2':
                from .gpt2 import load_weights
            elif c['model_type'] == 'olmoe':
                from .moe import load_weights
            elif c['model_type'] == 'starcoder2':
                from .starcoder import load_weights
            elif c["model_type"] == "gemma3_text":
                from .gemma import load_weights
            else:
                from .hybrid import load_weights
            self.finalize_weights(load_weights(state, c))
            return
        weights = {}
        def take(key):
            return state.pop(key).to(torch.float16)
        weights["embed"] = take("model.embed_tokens.weight")
        weights["norm"] = take("model.norm.weight")
        weights["head"] = weights["embed"] if c.get("tie_word_embeddings") else take("lm_head.weight")
        for i in range(c["num_hidden_layers"]):
            prefix = f"model.layers.{i}."
            if c['model_type'] == 'phi3':
                weights[f'{i}.qkv.weight'] = take(prefix + 'self_attn.qkv_proj.weight')
            else:
                for kind in (("weight", "bias") if c["model_type"] == "qwen2" else ("weight",)):
                    weights[f"{i}.qkv.{kind}"] = torch.cat([
                        take(prefix + f"self_attn.{p}_proj.{kind}") for p in ("q", "k", "v")
                    ])
            if c["model_type"] == "qwen3":
                weights[f"{i}.q_norm"] = take(prefix + "self_attn.q_norm.weight")
                weights[f"{i}.k_norm"] = take(prefix + "self_attn.k_norm.weight")
            weights[f"{i}.out"] = take(prefix + "self_attn.o_proj.weight")
            if c['model_type'] == 'phi3':
                weights[f'{i}.gate_up'] = take(prefix + 'mlp.gate_up_proj.weight')
            else:
                weights[f"{i}.gate_up"] = torch.cat([
                    take(prefix + f"mlp.{p}_proj.weight") for p in ("gate", "up")
                ])
            weights[f"{i}.down"] = take(prefix + "mlp.down_proj.weight")
            weights[f"{i}.norm1"] = take(prefix + "input_layernorm.weight")
            weights[f"{i}.norm2"] = take(prefix + "post_attention_layernorm.weight")
        # Keep host backing pageable: pinning whole multi-GB models can harm the
        # machine. Transfers are synchronous in this first executor.
        self.finalize_weights(weights)

    def finalize_weights(self, weights):
        from .quantization import prepare
        self.scales = prepare(weights, self.quantization)
        self.weights = weights
        self.bytes = sum(t.numel() * t.element_size() for t in
                         {id(t):t for t in (*weights.values(), *self.scales.values())}.values())

    def chat_tokens(self, messages, enable_thinking=None):
        if self.template is None:
            raise ValueError('model has no chat template; use raw completions')
        if not messages:
            raise ValueError("at least one message is required")
        for message in messages:
            if message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str):
                raise ValueError("only text system/user/assistant messages are supported")
        options = {}
        if enable_thinking is not None:
            if type(enable_thinking) is not bool or self.config["model_type"] not in ("qwen3", "qwen3_5_text"):
                raise ValueError("enable_thinking is a boolean option for Qwen3/Qwen3.5 models only")
            options["enable_thinking"] = enable_thinking
        rendered = self.template.render(messages=messages, add_generation_prompt=True,
                                        tools=None, **options, **self.special_tokens)
        ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        if set(ids) & set(self.config.get("multimodal_token_ids", [])):
            raise ValueError("multimodal input is not supported by the text executor")
        return ids

    def raw_tokens(self, prompt):
        from .prompts import validate_raw
        prompt = validate_raw(prompt)
        ids = self.tokenizer.encode(prompt, add_special_tokens=True).ids if isinstance(prompt, str) else prompt
        if not ids or any(token >= self.config['vocab_size'] for token in ids):
            raise ValueError('raw prompt must contain valid vocabulary token IDs')
        if set(ids) & set(self.config.get('multimodal_token_ids', [])):
            raise ValueError('multimodal input is not supported by the text executor')
        return ids


class Executor:
    def __init__(self, model, use_graph=True, fuse_projections=True, kernels=None, max_batch=1):
        self.device = selected_device()
        use_graph = use_graph and self.device == "cuda"
        self.model = model
        self.c = model.config
        self.context = model.max_context
        self.max_batch = max_batch
        self.fuse_projections = fuse_projections
        c = self.c
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.dim = c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.window = c.get("sliding_window") if c["model_type"] in ("phi3", "mistral", "starcoder2") else None
        self.precise = c["model_type"] in ("qwen3", "gemma3_text", "phi3", "starcoder2", "olmoe", "gpt2", "gpt_neox")
        self.rotary_precise = c["model_type"] in ("qwen3", "starcoder2", "olmoe")
        self.compute_dtype = torch.float32 if self.device == "mps" else torch.float16
        if c["model_type"] in ("gemma3_text", "starcoder2", "olmoe", "gpt2", "gpt_neox"):
            self.compute_dtype = torch.float32
        self.qwen3_precision = 'standard'
        if self.device == 'cuda' and c['model_type'] == 'qwen3':
            self.qwen3_precision = os.environ.get('DLLM_QWEN3_PRECISION', 'standard')
            if self.qwen3_precision not in ('standard', 'float32'):
                raise ValueError('DLLM_QWEN3_PRECISION must be standard or float32')
            if self.qwen3_precision != 'standard':
                self.compute_dtype = torch.float32
        self.dense_precision = self.qwen3_precision
        if self.device == 'cuda' and (c.get('rope_scaling') or {}).get('rope_type') == 'yarn':
            # The FP16 YaRN path fails the independent logit gate. Reuse the
            # qualified FP32 arithmetic path while retaining FP16 matrices.
            self.dense_precision = 'float32'
            self.compute_dtype = torch.float32
            self.precise = self.rotary_precise = True
        self.residual_dtype = torch.float32 if self.precise else self.compute_dtype
        selected = kernels or os.environ.get("DLLM_KERNELS", "auto")
        if selected not in ("auto", "torch", "triton", "metal"):
            raise ValueError("kernels must be auto, torch, triton, or metal")
        self.kernels = None
        if self.device == 'mps':
            if selected == 'triton':
                raise ValueError('Triton kernels require CUDA')
            if selected != 'torch':
                try:
                    from . import metal_kernels
                    self.kernels = metal_kernels
                except ImportError:
                    if selected == 'metal':
                        raise
        else:
            if selected == 'metal':
                raise ValueError('Metal kernels require MPS')
            if selected != 'torch':
                try:
                    from . import kernels as fused
                    self.kernels = fused
                except ImportError:
                    if selected == "triton":
                        raise
        if model.scales and self.kernels is None:
            raise ValueError('quantized weights require native CUDA kernels')
        self.linear = self.kernels.linear if self.device == 'mps' and self.kernels else F.linear
        half_metal_weights = False
        if self.device == 'mps':
            storage = os.environ.get('DLLM_METAL_WEIGHT_DTYPE', 'float32')
            if storage not in ('float16', 'float32'):
                raise ValueError('DLLM_METAL_WEIGHT_DTYPE must be float16 or float32')
            half_metal_weights = storage == 'float16'
            if half_metal_weights and self.kernels is None:
                raise ValueError('FP16 MPS weight storage requires native Metal kernels')
        half_gemma_weights = False
        if self.device == 'cuda' and c['model_type'] == 'gemma3_text':
            storage = os.environ.get('DLLM_GEMMA_WEIGHT_DTYPE', 'float16' if self.kernels else 'float32')
            if storage not in ('float16', 'float32'):
                raise ValueError('DLLM_GEMMA_WEIGHT_DTYPE must be float16 or float32')
            half_gemma_weights = storage == 'float16'
        kv_bytes = 2 * c["num_hidden_layers"] * max_batch * self.kv_heads * self.context * self.dim * (4 if self.compute_dtype == torch.float32 else 2)
        # A prior failed construction can leave unused allocator blocks cached
        # after its traceback is released. Return those before querying free VRAM.
        empty_cache()
        free = available_memory()
        def device_dtype(name, tensor):
            if tensor.dtype in (torch.int8,torch.uint8):
                return tensor.dtype
            if self.dense_precision != 'standard' and tensor.ndim == 2:
                return torch.float16
            if c['model_type'] == 'olmoe' and tensor.ndim == 2:
                return torch.float16
            if c['model_type'] == 'starcoder2' and 'norm' in name:
                return torch.float32
            if c['model_type'] == 'phi3' and name == 'norm':
                return torch.float32
            if c['model_type'] == 'gemma3_text' and (tensor.ndim == 1 or (self.device == 'mps' and name.endswith('.down'))):
                return torch.float32
            if (half_metal_weights or half_gemma_weights) and tensor.ndim == 2:
                return torch.float16
            return self.compute_dtype
        storage = {id(t): (t, device_dtype(name, t)) for name, t in model.weights.items()}
        device_weight_bytes = sum(t.numel() * torch.empty((), dtype=dtype).element_size()
                                  for t, dtype in storage.values())
        device_weight_bytes += sum(t.numel() * t.element_size() for t in model.scales.values())
        if device_weight_bytes + kv_bytes + 512 * 1024**2 > free:
            raise MemoryError("insufficient GPU memory for weights, KV cache, and workspace reserve")
        self.w = {}
        copies = {}
        for name, tensor in model.weights.items():
            if id(tensor) not in copies:
                dtype = device_dtype(name, tensor)
                copies[id(tensor)] = tensor.to(device=self.device, dtype=dtype)
            self.w[name] = copies[id(tensor)]
        if self.dense_precision != 'standard':
            self.linear = self.dense_linear
        self.quant_linear = None
        if model.scales:
            from .quantization import Linear
            self.quant_linear = Linear(self.w, model.scales, self.device, self.linear)
            self.linear = self.quant_linear
        self.kv = [torch.zeros((2, max_batch, self.kv_heads, self.context, self.dim),
                               dtype=self.compute_dtype, device=self.device)
                   for _ in range(c["num_hidden_layers"])]
        rotary_config = {**c, 'head_dim':c['rotary_dim']} if c['model_type']=='gpt_neox' else c
        inv = inverse_frequencies(rotary_config, self.device)
        angles = torch.outer(torch.arange(self.context, device=self.device, dtype=torch.float32), inv)
        angles = torch.cat((angles, angles), dim=-1)
        scale = attention_factor(c)
        self.cos, self.sin = (angles.cos()*scale).to(self.residual_dtype), (angles.sin()*scale).to(self.residual_dtype)
        self.arange = torch.arange(self.context, device=self.device)
        self.token = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.graph = None
        if self.device == "cuda" and (use_graph or self.kernels):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                try:
                    for _ in range(3):
                        self.forward(self.token, self.position, prefill=False)
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError) as error:
                    if selected != "auto" or self.kernels is None:
                        raise
                    self.kernels = None
                    print(f"dllm: fused kernel toolchain unavailable ({type(error).__name__}); using torch kernels", file=sys.stderr)
                    for _ in range(3):
                        self.forward(self.token, self.position, prefill=False)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        self.kernel_backend = ("metal" if self.device == "mps" else "triton") if self.kernels else "torch"
        if model.scales:
            self.kernel_backend += '+'+model.quantization
        if use_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.inference_mode():
                self.graph_logits = self.forward(self.token, self.position, prefill=False)
                self.graph_next = self.graph_logits.argmax(dim=-1)
        self.length = 0

    def dense_linear(self, x, weight, bias=None):
        if self.kernels and weight.dtype == torch.float16 and 3 <= x.shape[0] <= 128:
            from .precise_matmul import linear
            result = linear(x.float(), weight)
            return result if bias is None else result + bias.float()
        if self.kernels:
            result = self.kernels.linear_fp32(x.float(), weight)
            return result if bias is None else result + bias.float()
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float())

    def norm(self, x, weight):
        if self.kernels:
            return self.kernels.rms(x, weight, self.eps, self.precise)
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        if self.precise:
            return (normalized * weight.float()).to(weight.dtype)
        return normalized.to(weight.dtype) * weight

    def normalize_qk(self, q, k, layer):
        if self.c["model_type"] == "qwen3":
            def norm(x, weight):
                heads = x.reshape(-1, self.dim).float()
                return (heads * torch.rsqrt(heads.square().mean(-1, keepdim=True)+self.eps)
                        * weight.float()).reshape(x.shape)
            q, k = norm(q, self.w[f"{layer}.q_norm"]), norm(k, self.w[f"{layer}.k_norm"])
        return q, k

    def rotate(self, x, cos, sin):
        left, right = x.chunk(2, dim=-1)
        a, b = x * cos, torch.cat((-right, left), dim=-1) * sin
        if self.device == 'cuda' and self.c['model_type'] == 'phi3':
            # Keep the checkpoint's FP16 rotary multiplication boundaries even
            # though residual accumulation and final logits use FP32.
            a, b = a.half(), b.half()
        return (a + b).to(self.compute_dtype)

    def activate(self, gate, up):
        if self.precise:
            return (F.silu(gate.float()) * up.float()).to(self.compute_dtype)
        return F.silu(gate) * up

    def output_logits(self, x):
        normalized = self.norm(x, self.w['norm'])
        if self.c['model_type'] == 'phi3' and self.device == 'cuda':
            if self.w['head'].dtype in (torch.int8,torch.uint8):
                return self.quant_linear(normalized.float(), self.w['head'])
            if self.kernels:
                return self.kernels.linear_fp32(normalized.float(), self.w['head'])
            return F.linear(normalized.float(), self.w['head'].float())
        return self.linear(normalized, self.w['head'])

    def prefill_attention(self, query, key, value, query_start=0):
        if query_start:
            if (self.device=='cuda' and query.dtype==torch.float16
                    and (not self.window or key.shape[-2]<=self.window)):
                # A materialized offset mask disables fused Flash SDPA on
                # torch2.6. Its lower-right causal bias preserves the kernel.
                from torch.nn.attention.bias import causal_lower_right
                return scaled_attention(query,key,value,
                    attn_mask=causal_lower_right(query.shape[-2],key.shape[-2]),enable_gqa=True)
            qpos=self.arange[query_start:query_start+query.shape[-2],None]
            kpos=self.arange[None,:key.shape[-2]]
            mask=kpos<=qpos
            if self.window:
                mask=mask & (kpos>qpos-self.window)
            return scaled_attention(query,key,value,attn_mask=mask,enable_gqa=True)
        if not self.window:
            return scaled_attention(query, key, value, is_causal=True, enable_gqa=True)
        if query.shape[-2] > 2048:
            # Masked GQA can select the quadratic-memory SDPA math backend.
            # Bound its query dimension while retaining the exact global window.
            outputs = []
            for start in range(0, query.shape[-2], 256):
                end = min(start+256, query.shape[-2])
                first_key = max(0, start-self.window+1)
                qpos = self.arange[start:end, None]
                kpos = self.arange[None, first_key:end]
                mask = (kpos <= qpos) & (kpos > qpos-self.window)
                outputs.append(scaled_attention(query[:,:,start:end], key[:,:,first_key:end],
                    value[:,:,first_key:end], attn_mask=mask, enable_gqa=True))
            return torch.cat(outputs, dim=-2)
        positions = self.arange[:query.shape[-2]]
        mask = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - self.window)
        return scaled_attention(query, key, value, attn_mask=mask, enable_gqa=True)

    def decode_attention(self, query, cache_k, cache_v, positions, slots=None):
        if self.kernels and (not self.window or self.device == 'cuda'):
            options = {'window': self.window} if self.window else {}
            return self.kernels.attention(query, cache_k, cache_v, positions, slots, **options)
        mask = self.arange[None, None, None, :] <= positions[:, None, None, None]
        if self.window:
            mask = mask & (self.arange[None, None, None, :] > positions[:, None, None, None] - self.window)
        key, value = (cache_k, cache_v) if slots is None else (cache_k.index_select(0, slots), cache_v.index_select(0, slots))
        return scaled_attention(query, key, value, attn_mask=mask, enable_gqa=True)

    @torch.inference_mode()
    def forward(self, ids, positions, prefill, slot=0):
        w = self.w
        x = F.embedding(ids, w["embed"]).to(self.residual_dtype)
        count = ids.shape[0]
        if not self.kernels or not self.fuse_projections:
            cos = self.cos[positions].view(1, 1, count, self.dim)
            sin = self.sin[positions].view(1, 1, count, self.dim)
        for i in range(self.c["num_hidden_layers"]):
            hidden = self.norm(x, w[f"{i}.norm1"])
            sizes = [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim]
            cache_k, cache_v = self.kv[i][:, slot:slot+1].unbind(0)
            if self.fuse_projections:
                qkv = self.linear(hidden, w[f"{i}.qkv.weight"], w.get(f"{i}.qkv.bias"))
                q, k, v = qkv.split(sizes, dim=-1)
            else:
                bias = w.get(f"{i}.qkv.bias")
                biases = bias.split(sizes) if bias is not None else (None, None, None)
                q, k, v = [self.linear(hidden, weight, part_bias) for weight, part_bias in zip(
                    w[f"{i}.qkv.weight"].split(sizes), biases)]
            if self.kernels and self.fuse_projections:
                q = self.kernels.rope_store(qkv, self.cos, self.sin, positions, cache_k, cache_v, self.heads,
                    w.get(f"{i}.q_norm"), w.get(f"{i}.k_norm"), self.eps, self.rotary_precise)
                k, v = cache_k[:, :, :count], cache_v[:, :, :count]
            else:
                q, k = self.normalize_qk(q, k, i)
                q = self.rotate(q.view(1, count, self.heads, self.dim).transpose(1, 2), cos, sin)
                k = self.rotate(k.view(1, count, self.kv_heads, self.dim).transpose(1, 2), cos, sin)
                v = v.view(1, count, self.kv_heads, self.dim).transpose(1, 2)
                if self.device == 'mps':
                    cache_k[:, :, positions, :] = k
                    cache_v[:, :, positions, :] = v
                else:
                    cache_k.index_copy_(2, positions, k)
                    cache_v.index_copy_(2, positions, v)
            if not prefill:
                k, v = cache_k, cache_v
            attention = (self.prefill_attention(q, k, v) if prefill
                         else self.decode_attention(q, cache_k, cache_v, positions))
            x = x + self.linear(attention.transpose(1, 2).reshape(count, -1), w[f"{i}.out"])
            hidden = self.norm(x, w[f"{i}.norm2"])
            if self.fuse_projections:
                gate_up = self.linear(hidden, w[f"{i}.gate_up"])
                gate, up = gate_up.chunk(2, -1)
            else:
                gate, up = [self.linear(hidden, weight) for weight in w[f"{i}.gate_up"].chunk(2)]
            activated = self.kernels.swiglu(gate_up, self.precise) if self.kernels and self.fuse_projections else self.activate(gate, up)
            x = x + self.linear(activated, w[f"{i}.down"])
        return self.output_logits(x[-1:])

    @torch.inference_mode()
    def prefill(self, tokens, slot=0):
        if not tokens or len(tokens) > self.context:
            raise ValueError("prompt is empty or exceeds context")
        if not 0 <= slot < self.max_batch:
            raise ValueError("invalid cache slot")
        ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
        self.length = len(tokens)
        return self.forward(ids, torch.arange(len(tokens), device=self.device), prefill=True, slot=slot)

    @torch.inference_mode()
    def decode(self, token):
        if self.length >= self.context:
            raise ValueError("context exhausted")
        self.token.fill_(token)
        self.position.fill_(self.length)
        self.length += 1
        if self.graph is not None:
            self.graph.replay()
            return self.graph_logits, self.graph_next
        logits = self.forward(self.token, self.position, prefill=False)
        return logits, logits.argmax(dim=-1)
