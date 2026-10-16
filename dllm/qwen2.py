"""Independent dense decoder using PyTorch/CUDA and dllm kernels, not HF generation.

Dense Qwen2, Qwen3, Llama and Phi-3; model-specific mixed precision.
The single-sequence base is extended by BatchExecutor. CUDA graphs remove
per-operation Python dispatch within a captured decode step; the host still
stages metadata, replays the graph, and reads the selected token.
"""

from contextlib import closing
from io import StringIO
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from .device import selected_device, empty_cache, available_memory, attention as scaled_attention
from safetensors import safe_open
from tokenizers import Tokenizer
from .chat_template import compile_template, RENDER_ERRORS

from .tokenizer_metadata import read_tokenizer_config, read_generation_config, special_token_text, checkpoint_template
from .config import decoder_config, eos_ids, read_config
from .rotary import inverse_frequencies, attention_factor


MAX_RENDERED_CHAT_BYTES = 16 * 1024 * 1024


def weight_files(path):
    from .checkpoint import inspect_snapshot
    return inspect_snapshot(path)[0]


def weight_bytes(path,quantization=None):
    """Estimate host weight storage from headers before loading tensors."""
    from .quantization import mode
    quantization=mode(quantization)
    quantized = quantization != 'none'
    config = decoder_config(read_config(Path(path) / 'config.json'))
    from .vision_policy import enabled as vision_enabled,eligible as vision_eligible
    include_vision=vision_eligible(config) and vision_enabled()
    if quantization == 'q4_0' and config['model_type'] == 'qwen3_5_text':
        from .hybrid_q4_policy import validate
        validate(config)
    elif quantization=='q4_0' and config['model_type'] not in ('qwen2','qwen3','llama','phi3','mistral'):
        raise ValueError('Q4_0 is currently supported only for dense rotary decoders')
    if config['model_type'] in ('gpt2','gpt_neox','bloom') and quantized:
        raise ValueError('GPT-2/GPT-NeoX/BLOOM quantization is not implemented')
    if config['model_type']=='gpt_oss' and quantized:
        raise ValueError('GPT-OSS uses checkpoint-native MXFP4; conversion is unsupported')
    element_bytes = 4 if config['model_type'] in ('starcoder2','gpt2','gpt_neox','bloom') else 2
    total, embedding, has_head = 0, None, False
    for file in weight_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith('mtp.'):
                    continue
                if key.startswith('model.visual.'):
                    if include_vision:
                        shape=reader.get_slice(key).get_shape()
                        projection=key.endswith('.weight') and key!='model.visual.pos_embed.weight' and '.norm' not in key
                        total+=math.prod(shape)*(2 if projection else 4)
                    continue
                if config['model_type']=='gpt2' and key.endswith(('attn.bias','attn.masked_bias')):
                    continue
                if config['model_type']=='gpt_neox' and key.endswith(('attention.bias','attention.masked_bias','rotary_emb.inv_freq')):
                    continue
                shape = reader.get_slice(key).get_shape()
                is_embedding = key.endswith('embed_tokens.weight')
                if is_embedding:
                    embedding = shape
                has_head = has_head or key == 'lm_head.weight'
                if config['model_type']=='gpt_oss':
                    total += math.prod(shape)*(1 if key.endswith(('_blocks','_scales')) else
                        4 if len(shape)==1 or key.endswith('_bias') or reader.get_slice(key).get_dtype()=='F32' else 2)
                    continue
                if quantized and len(shape) == 2 and not is_embedding and shape[1] % 32 == 0:
                    total += math.prod(shape)//(2 if quantization=='q4_0' else 1) + 2 * shape[0] * (shape[1] // 32)
                else:
                    tensor_bytes=4 if config['model_type']=='qwen3_5_text' and len(shape)==1 else element_bytes
                    total += tensor_bytes * math.prod(shape)
    if quantized and embedding is not None and not has_head:
        if config.get('tie_word_embeddings'):
            total += math.prod(embedding)//(2 if quantization=='q4_0' else 1) + 2 * embedding[0] * (embedding[1] // 32)
    return total



class ModelWeights:
    def __init__(self, path, max_context, quantization=None):
        from .quantization import mode
        self.quantization = mode(quantization)
        self.path = Path(path).resolve()
        self.config = decoder_config(read_config(self.path / "config.json"))
        c = self.config
        if selected_device() == 'xpu':
            from .xpu_policy import validate_model
            validate_model(c, self.quantization, max_context)
        self.weight_format = 'mxfp4' if c['model_type']=='gpt_oss' else self.quantization
        if c['model_type']=='gpt_oss' and self.quantization!='none':
            raise ValueError('GPT-OSS uses checkpoint-native MXFP4; conversion is unsupported')
        if self.quantization != 'none' and selected_device() != 'cuda':
            if self.quantization == 'q4_0':
                from .metal_q4_policy import validate as validate_metal_q4
                validate_metal_q4(c, max_context)
            else:
                from .hybrid_metal import eligible as hybrid_eligible
                supported=c['model_type'] in ('qwen2','qwen3','llama') or hybrid_eligible(c,max_context)
                if self.quantization!='q8_0' or not supported:
                    raise ValueError('Metal decoder quantization supports Q8 dense Qwen2/Qwen3/Llama and qualified Qwen3.5-0.8B only')
        if self.quantization == 'q4_0' and c['model_type'] == 'qwen3_5_text':
            from .hybrid_q4_policy import validate
            validate(c, max_context)
        elif self.quantization=='q4_0' and c['model_type'] not in ('qwen2','qwen3','llama','phi3','mistral'):
            raise ValueError('Q4_0 is currently supported only for dense rotary decoders')
        if c['model_type'] in ('gpt2','gpt_neox','bloom') and self.quantization!='none':
            raise ValueError('GPT-2/GPT-NeoX/BLOOM quantization is not implemented')
        if not 1 <= max_context <= c["max_position_embeddings"]:
            raise ValueError("context exceeds model position limit")
        files = weight_files(self.path)
        from .layer_stream import enabled as offload_enabled, validate as validate_offload
        offload = offload_enabled()
        if offload:
            validate_offload(c, self.quantization, max_context)
            from .host_memory import admit_checkpoint, require_headroom
            admit_checkpoint(files)
        self.max_context = max_context
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        tokenizer_config = read_tokenizer_config(self.path / "tokenizer_config.json")
        template = checkpoint_template(self.path, tokenizer_config)
        self.configure_tokenizer(self.tokenizer, template, tokenizer_config,
                                 tool_template=checkpoint_template(self.path, tokenizer_config, tools=True))
        self.eos = eos_ids(c.get('eos_token_id'))
        generation = self.path / "generation_config.json"
        if generation.exists():
            saved=read_generation_config(generation)
            if 'eos_token_id' in saved:self.eos=eos_ids(saved['eos_token_id'])
        if not self.eos:
            raise ValueError('model needs at least one EOS token ID')
        if any(token >= c['vocab_size'] for token in self.eos):
            raise ValueError('EOS token IDs must be within the model vocabulary')
        from .weight_cache import configured_cache
        cache = configured_cache(self.path, c, self.quantization, files)
        self.weight_cache_status = 'disabled' if cache is None else 'miss'
        if cache is not None:
            prepared = cache.load()
            if prepared is not None:
                self.finalize_weights(prepared[0], prepared[1])
                self.weight_cache_status = 'hit'
                return
        state = {}
        for candidate in files:
            with safe_open(str(candidate), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if not key.startswith(('model.visual.', 'mtp.')):
                        if offload:
                            # Recheck before faulting in each source tensor.
                            header = reader.get_slice(key)
                            require_headroom(math.prod(header.get_shape()) *
                                             (4 if header.get_dtype() == 'F32' else 2))
                        state[key] = reader.get_tensor(key)
        if c["model_type"] in ("qwen3_5_text", "gemma3_text", "starcoder2", "olmoe", "gpt2", "gpt_neox", "bloom", "gpt_oss"):
            if c['model_type'] == 'gpt_oss':
                from .gpt_oss import load_weights
            elif c['model_type'] == 'bloom':
                from .bloom import load_weights
            elif c['model_type'] == 'gpt_neox':
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
        from .dense_checkpoint import validate as validate_dense_checkpoint
        validate_dense_checkpoint(state,c)
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
        if cache is not None:
            self.weight_cache_status = 'stored' if cache.save(self.weights, self.scales) else 'write-failed'

    def configure_tokenizer(self, tokenizer, template, tokenizer_config, *, tool_template=None):
        """Share template behavior across HF snapshots and standalone GGUF files."""
        self.tokenizer = tokenizer
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        c = self.config
        if template is not None and not isinstance(template, str):
            raise ValueError("model needs one text chat template")
        self.template = compile_template(template) if template is not None else None
        if tool_template is not None and not isinstance(tool_template, str):
            raise ValueError('model needs one text tool template')
        self.tool_template = (compile_template(tool_template)
            if tool_template is not None and tool_template != template else self.template)
        tool_source = template if tool_template is None else tool_template
        self.supports_tool_calls=(c['model_type'] in ('qwen2','qwen3') and isinstance(tool_source,str)
            and all(marker in tool_source for marker in ('<tool_call>','</tool_call>','tool_calls','tools')))
        self.supports_tool_calls |= (c['model_type']=='gpt_oss' and isinstance(tool_source,str)
            and all(marker in tool_source for marker in ('functions.', '<|call|>', 'tool_calls', 'tools')))
        self.special_tokens = special_token_text(tokenizer_config)

    def finalize_weights(self, weights, prepared_scales=None):
        from .quantization import prepare
        self.scales = prepare(weights, self.quantization) if prepared_scales is None else prepared_scales
        self.weights = weights
        self.bytes = sum(t.numel() * t.element_size() for t in
                         {id(t):t for t in (*weights.values(), *self.scales.values())}.values())

    def chat_tokens(self, messages, enable_thinking=None, tools=None, tool_json=False, reasoning_effort=None):
        if self.template is None:
            raise ValueError('model has no chat template; use raw completions')
        from .tool_calls import template_messages,validate_tools
        messages=template_messages(messages)
        tools=validate_tools(tools)
        tool_request = bool(tools) or any(m['role']=='tool' or m.get('tool_calls') for m in messages)
        if tool_request:
            if not self.supports_tool_calls:
                raise ValueError('tool chat requires a qualified function-call template')
        template = getattr(self, 'tool_template', self.template) if tool_request else self.template
        if self.config['model_type']=='gpt_oss':
            if tool_json:raise ValueError('Harmony supports auto or none tool choice only')
            # The pinned template expects non-null content, parsed arguments and
            # one function per message. Never silently discard extra calls.
            for message in messages:
                calls=message.get('tool_calls')
                if calls:
                    if len(calls)!=1:raise ValueError('Harmony history supports one call per assistant message')
                    message['content']=message.get('content') or ''
                    if message.get('reasoning_content'):
                        if message['content']:
                            raise ValueError('Harmony tool history cannot combine commentary and reasoning')
                        message['thinking']=message['reasoning_content']
                elif message.get('role')=='assistant':message.pop('tool_calls',None)
            for tool in tools or []:tool['function'].setdefault('description','')
        if tool_json and not tools:raise ValueError('forced tool prefix requires tools')
        options = {}
        if reasoning_effort is not None:
            if (self.config['model_type']!='gpt_oss' or not isinstance(reasoning_effort,str)
                    or reasoning_effort not in ('low','medium','high')):
                raise ValueError('reasoning_effort requires GPT-OSS and low, medium or high')
            options['reasoning_effort']=reasoning_effort
        if enable_thinking is not None:
            if type(enable_thinking) is not bool or self.config["model_type"] not in ("qwen3", "qwen3_5_text"):
                raise ValueError("enable_thinking is a boolean option for Qwen3/Qwen3.5 models only")
            options["enable_thinking"] = enable_thinking
        suffix = '<tool_call>\n' if tool_json else ''
        size = len(suffix.encode('utf-8'))
        if size > MAX_RENDERED_CHAT_BYTES:
            raise ValueError('chat rendered prompt exceeds 16 MiB')
        try:
            with StringIO() as output, closing(template.generate(messages=messages, add_generation_prompt=True,
                                                tools=tools, **options, **self.special_tokens)) as stream:
                for chunk in stream:
                    size += len(chunk.encode('utf-8'))
                    if size > MAX_RENDERED_CHAT_BYTES:
                        raise ValueError('chat rendered prompt exceeds 16 MiB')
                    output.write(chunk)
                rendered = output.getvalue() + suffix
        except RENDER_ERRORS as error:
            raise ValueError('chat template rendering failed') from error
        ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        if not ids or any(token >= self.config['vocab_size'] for token in ids):
            raise ValueError('chat prompt must contain valid vocabulary token IDs')
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
        if self.device == 'xpu':
            from .xpu_policy import validate_model
            validate_model(model.config, model.quantization, model.max_context, max_batch)
        if self.device == 'mps' and model.quantization == 'q4_0':
            from .metal_q4_policy import validate as validate_metal_q4
            validate_metal_q4(model.config, model.max_context, max_batch)
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
        self.precise = c["model_type"] in ("qwen3", "gemma3_text", "phi3", "starcoder2", "olmoe", "gpt2", "gpt_neox", "bloom")
        self.rotary_precise = c["model_type"] in ("qwen3", "starcoder2", "olmoe")
        self.compute_dtype = torch.float32 if self.device in ("mps", "xpu") else torch.float16
        if c["model_type"] in ("gemma3_text", "starcoder2", "olmoe", "gpt2", "gpt_neox", "bloom"):
            self.compute_dtype = torch.float32
        self.qwen3_precision = 'standard'
        if self.device == 'cuda' and c['model_type'] == 'qwen3':
            self.qwen3_precision = os.environ.get('DLLM_QWEN3_PRECISION', 'standard')
            if self.qwen3_precision not in ('standard', 'float32'):
                raise ValueError('DLLM_QWEN3_PRECISION must be standard or float32')
            if self.qwen3_precision != 'standard':
                self.compute_dtype = torch.float32
        self.dense_precision = self.qwen3_precision
        self.llama_precision = 'standard'
        if self.device == 'cuda' and c['model_type'] == 'llama':
            self.llama_precision = os.environ.get('DLLM_LLAMA_PRECISION','mixed')
            if self.llama_precision not in ('standard','mixed','float32'):
                raise ValueError('DLLM_LLAMA_PRECISION must be standard, mixed or float32')
            if self.llama_precision in ('mixed','float32'):
                # Both policies avoid repeated FP16 residual accumulation.
                # Mixed retains Tensor Core projections and FP16 KV.
                self.precise = self.rotary_precise = True
            if self.llama_precision == 'float32':
                self.dense_precision = 'float32'
                self.compute_dtype = torch.float32
        if self.device == 'cuda' and (c.get('rope_scaling') or {}).get('rope_type') == 'yarn':
            # The FP16 YaRN path fails the independent logit gate. Reuse the
            # qualified FP32 arithmetic path while retaining FP16 matrices.
            self.dense_precision = 'float32'
            self.compute_dtype = torch.float32
            self.precise = self.rotary_precise = True
        if getattr(model, 'awq_fp16_rounding', False):
            if c['model_type'] != 'qwen3' or not getattr(model, 'awq_matrices', {}) or self.compute_dtype != torch.float16:
                raise ValueError('AWQ FP16 rounding qualification requires Qwen3 FP16 AWQ')
            self.precise = self.rotary_precise = False
        self.residual_dtype = torch.float32 if self.precise else self.compute_dtype
        selected = kernels or os.environ.get("DLLM_KERNELS", "auto")
        if selected not in ("auto", "torch", "triton", "metal"):
            raise ValueError("kernels must be auto, torch, triton, or metal")
        self.kernels = None
        if self.device == 'xpu':
            if selected not in ('auto', 'torch'):
                raise ValueError('Intel XPU currently requires eager torch kernels')
        elif self.device == 'mps':
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
        packed_formats = getattr(model, 'packed_formats', {})
        awq_matrices = getattr(model, 'awq_matrices', {})
        if awq_matrices and (self.device != 'cuda' or self.compute_dtype != torch.float16
                             or not self.fuse_projections or self.kernels is None):
            raise ValueError('AWQ requires fused native CUDA execution with FP16 matrices')
        if packed_formats and (self.device != 'cuda' or not self.fuse_projections):
            raise ValueError('native packed layouts require fused CUDA execution')
        if (model.scales or packed_formats) and self.kernels is None:
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
        kv_bytes = 2 * c["num_hidden_layers"] * getattr(self, 'kv_token_capacity', max_batch * self.context) * self.kv_heads * self.dim * (4 if self.compute_dtype == torch.float32 else 2)
        # A prior failed construction can leave unused allocator blocks cached
        # after its traceback is released. Return those before querying free VRAM.
        empty_cache()
        free = available_memory()
        def device_dtype(name, tensor):
            if name in awq_matrices:
                return torch.int32
            if name == 'rope_factors':
                return torch.float32
            if c['model_type']=='gpt_oss':
                return tensor.dtype
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
        device_weight_bytes += sum(t.numel() * t.element_size() for t in {id(t):t for t in model.scales.values()}.values())
        device_weight_bytes += sum(t.numel() * t.element_size() for t in
                                   {id(m.zeros): m.zeros for m in awq_matrices.values()}.values())
        self.packed_workspace_bytes = max((width*spec[-1][1]*(4 if self.compute_dtype==torch.float32 else 2)
            for name,(width,spec) in packed_formats.items() if name!='embed'),default=0)
        self.packed_workspace_bytes = max(self.packed_workspace_bytes,
                                         max((m.inputs*m.outputs*2 for m in awq_matrices.values()), default=0))
        native_awq = None
        if awq_matrices:
            from .native_awq import select as select_awq, workspace_bytes as awq_workspace_bytes
            native_awq = select_awq(c, self.device, self.compute_dtype, self.context, max_batch)
            if native_awq is not None:
                # Repacking is sequential after transfer aliases are released.
                # Existing dense-projection reserve covers one packed replacement
                # and its zero/scale permutation temporaries; scratch is additional.
                self.packed_workspace_bytes += awq_workspace_bytes(native_awq)
        from .native_q4_marlin import select as select_q4_marlin
        native_q4_marlin, q4_marlin_plan = select_q4_marlin(
            c, model.weights, model.scales, model.quantization, self.device,
            self.compute_dtype, self.llama_precision, self.context, max_batch,
            self.fuse_projections and self.kernels is not None and not packed_formats and not awq_matrices)
        self.packed_layout_overhead_bytes = q4_marlin_plan.overhead_bytes
        self.q4_marlin_abi = native_q4_marlin.abi if native_q4_marlin is not None else None
        device_weight_bytes += self.packed_layout_overhead_bytes
        if native_q4_marlin is not None:
            self.packed_workspace_bytes += q4_marlin_plan.repack_peak_bytes + native_q4_marlin.workspace_bytes
        self.required_memory_bytes = device_weight_bytes + kv_bytes + self.packed_workspace_bytes + 512 * 1024**2
        if self.required_memory_bytes > free:
            raise MemoryError("insufficient GPU memory for weights, KV cache, and workspace reserve")
        from .kquant_cache import plan as plan_cache, build as build_cache
        cache_plan = plan_cache(c, model.weights, packed_formats, self.device, self.compute_dtype,
                               self.qwen3_precision if c['model_type'] == 'qwen3' else self.llama_precision, self.context,
                               free - device_weight_bytes - kv_bytes - self.packed_workspace_bytes - 512 * 1024**2)
        self.packed_cache_bytes = cache_plan.bytes
        self.w = {}
        copies = {}
        for name, tensor in model.weights.items():
            if id(tensor) not in copies:
                dtype = device_dtype(name, tensor)
                copies[id(tensor)] = tensor.to(device=self.device, dtype=dtype)
            self.w[name] = copies[id(tensor)]
        copies.clear()
        if self.dense_precision != 'standard':
            self.linear = self.dense_linear
        self.quant_linear = None
        self.packed_decode = None
        self.packed_prefill = None
        self.q4_decode = None
        if model.scales or packed_formats:
            from .quantization import Linear
            from .native_kquant_residual import select
            self.packed_decode = select(c, packed_formats, self.device, self.compute_dtype, self.llama_precision, self.context)
            if self.packed_decode is None:
                from .native_kquant_residual_f32 import select as select_fp32
                self.packed_decode = select_fp32(c, packed_formats, self.device, self.compute_dtype,
                                                self.llama_precision, self.context)
            from .kquant_prefill import select as select_prefill
            self.packed_prefill = select_prefill(c, packed_formats, self.device, self.compute_dtype,
                                                self.llama_precision, self.context)
            from .native_q4_residual import select as select_q4
            if native_q4_marlin is None:
                self.q4_decode = select_q4(c, model.quantization, self.device, self.compute_dtype,
                                          self.llama_precision, self.context)
            self.quant_linear = Linear(self.w, model.scales, self.device, self.linear, packed_formats,
                                       self.packed_decode, self.packed_prefill, self.q4_decode,
                                       packed_cache=build_cache(cache_plan, self.w, packed_formats), awq_matrices=awq_matrices, native_awq=native_awq,
                                       native_q4_marlin=native_q4_marlin, q4_marlin_plan=q4_marlin_plan)
            self.linear = self.quant_linear
        self.kv = self.allocate_kv()
        rotary_config = {**c, 'head_dim':c['rotary_dim']} if c['model_type']=='gpt_neox' else c
        inv = inverse_frequencies(rotary_config, self.device)
        if 'rope_factors' in self.w:
            inv = inv / self.w['rope_factors']
        angles = torch.outer(torch.arange(self.context, device=self.device, dtype=torch.float32), inv)
        angles = torch.cat((angles, angles), dim=-1)
        scale = attention_factor(c)
        self.cos, self.sin = (angles.cos()*scale).to(self.residual_dtype), (angles.sin()*scale).to(self.residual_dtype)
        self.arange = torch.arange(self.context, device=self.device)
        self.token = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.graph = None
        self.rotary_store = self.kernels.rope_store if self.kernels else None
        if self.device == "cuda" and (use_graph or self.kernels):
            from .cuda_warmup import warmup_stream
            with warmup_stream(), torch.inference_mode():
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
            torch.cuda.synchronize()
        self.kernel_backend = ("metal" if self.device == "mps" else "triton") if self.kernels else "torch"
        if model.scales or packed_formats:
            self.kernel_backend += '+'+('awq-marlin' if native_awq is not None else
                                        'awq-gemm' if awq_matrices else model.quantization)
            if self.packed_cache_bytes:
                self.kernel_backend += '+resident-kquant-cache'
            if native_q4_marlin is not None:
                self.kernel_backend += '+cuda-q4-marlin'
            if self.q4_decode is not None:
                self.kernel_backend += '+cuda-q4-residual'
            if self.packed_decode is not None:
                self.kernel_backend += ('+cuda-kquant-residual-f32' if self.compute_dtype == torch.float32
                                        else '+cuda-kquant-residual')
            if self.packed_prefill is not None:
                self.kernel_backend += '+compensated-kquant-prefill-f32'
            if (self.compute_dtype==torch.float16 and os.environ.get('DLLM_KQUANT_CUDA_LIBRARY')
                    and any(name!='embed' and all(kind in (12,13,14) for _,_,kind,_ in layout[1])
                            for name,layout in packed_formats.items())):
                self.kernel_backend += '+cuda-kquant-gemv'
            if (model.quantization=='q4_0' and self.compute_dtype==torch.float16
                    and os.environ.get('DLLM_Q4_CUDA_LIBRARY')
                    and any(w.ndim==2 and w.dtype==torch.uint8 and w.shape[0]>=1024 and w.shape[1]>=2048 for w in self.w.values())):
                self.kernel_backend += '+cuda-gemv'
        if use_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.graph_logits = self.forward(self.token, self.position, prefill=False)
                self.graph_next = self.graph_logits.argmax(dim=-1)
        self.length = 0
        self.configure_prefill_graphs(use_graph)

    def configure_prefill_graphs(self,use_graph):
        from .prefill_graphs import configuration,PrefillGraphs
        count,tokens=configuration()
        eligible=(use_graph and self.device=='cuda' and self.c['model_type'] in ('qwen2','qwen3')
            and not self.model.scales and not getattr(self.model,'packed_formats',{}) and not hasattr(self,'pool'))
        self.prefill_graph_cache=PrefillGraphs(count,tokens) if eligible and count else None

    def allocate_kv(self):
        return [torch.zeros((2, self.max_batch, self.kv_heads, self.context, self.dim),
                            dtype=self.compute_dtype, device=self.device)
                for _ in range(self.c['num_hidden_layers'])]

    def embedding(self, ids):
        weight = self.w['embed']
        if weight.dtype in (torch.int8, torch.uint8):
            return self.quant_linear.embedding(ids, weight, self.residual_dtype)
        return F.embedding(ids, weight).to(self.residual_dtype)

    def dense_linear(self, x, weight, bias=None):
        # Reuse compensated products at prefill sizes instead of expanding
        # every stored FP16 matrix to FP32 for each layer.
        if self.kernels and weight.dtype == torch.float16 and x.shape[0] >= 3:
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
        variance = xf.square().mean(-1, keepdim=True)
        # Finite inputs can overflow here and otherwise silently normalize to zero.
        if not torch.isfinite(variance).all():raise RuntimeError('nonfinite normalization variance')
        normalized = xf * torch.rsqrt(variance + self.eps)
        if self.precise:
            return (normalized * weight.float()).to(weight.dtype)
        result = normalized.to(weight.dtype) * weight
        if not torch.isfinite(result.float()).all():raise RuntimeError('nonfinite normalization output')
        return result

    def normalize_qk(self, q, k, layer):
        if self.c["model_type"] == "qwen3":
            def norm(x, weight):
                heads = x.reshape(-1, self.dim).float()
                variance = heads.square().mean(-1, keepdim=True)
                if not torch.isfinite(variance).all():raise RuntimeError('nonfinite QK normalization variance')
                result = (heads * torch.rsqrt(variance+self.eps)
                          * weight.float()).reshape(x.shape)
                if not torch.isfinite(result.float()).all():raise RuntimeError('nonfinite QK normalization output')
                return result
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
        if getattr(self,'_candidate_head',None) is not None:
            return self.linear(normalized,self._candidate_head)
        if self.c['model_type'] == 'phi3' and self.device == 'cuda':
            if self.w['head'].dtype in (torch.int8,torch.uint8):
                return self.quant_linear(normalized.float(), self.w['head'])
            if self.kernels:
                return self.kernels.linear_fp32(normalized.float(), self.w['head'])
            return F.linear(normalized.float(), self.w['head'].float())
        return self.linear(normalized, self.w['head'])

    @torch.inference_mode()
    def score_candidates(self,tokens,candidates):
        # Runtime owns an exclusive executor lease. Only eager prefill observes
        # this temporary head; captured decode graphs retain the complete head.
        # Prefill graphs retain selected heads only under their exact candidate key.
        if self.c['model_type'] not in ('qwen2','qwen3') or self.w['head'].dtype not in (torch.float16,torch.float32):
            raise ValueError('direct scoring requires unquantized Qwen2 or Qwen3')
        if any(token>=self.w['head'].shape[0] for token in candidates):
            raise ValueError('candidate token is outside the model vocabulary')
        if getattr(self,'_candidate_head',None) is not None:raise RuntimeError('nested candidate scoring')
        ids=torch.tensor(candidates,device=self.device,dtype=torch.long)
        self._candidate_head=self.w['head'].index_select(0,ids)
        self._candidate_ids=tuple(candidates)
        try:
            return self.prefill(tokens).float()
        finally:
            self._candidate_head=None
            self._candidate_ids=None

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
        hidden=self.forward_hidden(self.embedding(ids),positions,prefill,slot)
        return self.output_logits(hidden[-1:])

    @torch.inference_mode()
    def forward_hidden(self, x, positions, prefill, slot=0):
        w = self.w
        count = x.shape[0]
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
                q = self.rotary_store(qkv, self.cos, self.sin, positions, cache_k, cache_v, self.heads,
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
        return x

    @torch.inference_mode()
    def prefill(self, tokens, slot=0):
        if not tokens or len(tokens) > self.context:
            raise ValueError("prompt is empty or exceeds context")
        if not 0 <= slot < self.max_batch:
            raise ValueError("invalid cache slot")
        ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
        self.length = len(tokens)
        cache=self.prefill_graph_cache
        head=getattr(self,'_candidate_head',None)
        head_key=getattr(self,'_candidate_ids',None)
        if (cache is not None and len(tokens)<=cache.max_tokens
                and (head is None or head_key is not None)):
            return cache.run(ids,slot,self.forward,head_key=head_key,head=head)
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
