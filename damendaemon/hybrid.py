"""Dense Qwen3.5 text execution, independent of Transformers and vLLM.

Implements the published Gated DeltaNet recurrence, causal depthwise convolution,
zero-centered RMSNorm, and gated partial-rotary full attention. Vision, MoE, and
multi-token prediction are intentionally excluded from this text execution path.
"""
import ctypes
import os
import subprocess
import sys

import torch
from torch.nn import functional as F

from .device import selected_device, selected_accelerator, synchronize, empty_cache, available_memory, attention as scaled_attention

from .batch import BatchExecutor
from .token_gate import TokenGate


def _l2_normalize(value):
    """FP32 L2 normalization that rejects overflow instead of silently zeroing.

    A finite head whose square-sum overflows yields an infinite norm and an
    all-zero normalized vector, corrupting linear-attention geometry.
    """
    fp32 = value.float()
    norm = fp32.square().sum(-1, keepdim=True)
    if not torch.isfinite(norm).all():
        raise RuntimeError('nonfinite hybrid linear-attention q/k norm')
    return fp32 * torch.rsqrt(norm + 1.e-6)


def load_weights(state, config):
    # Multimodal checkpoints wrap the text backbone in language_model.
    prefix = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in state else 'model.'
    from .hybrid_checkpoint_contract import validate
    validate(state,config,prefix)
    w = {}
    from .layer_stream import enabled as offload_enabled
    from .host_memory import require_headroom
    guarded = offload_enabled() and any(t.device.type == 'cpu' for t in state.values())
    def convert(tensor, dtype):
        if guarded:
            # Two copies also cover a following norm offset allocation.
            require_headroom(tensor.numel() * 2 * dtype.itemsize)
        return tensor.to(dtype)
    def concatenate(tensors):
        if guarded:
            require_headroom(sum(t.numel() * t.element_size() for t in tensors))
        return torch.cat(tensors)
    def take(name, dtype=torch.float16):
        from .device import selected_accelerator
        if state[prefix + name].device.type != 'meta' and selected_accelerator() == 'rocm':
            from .rocm_policy import require_host_headroom
            require_host_headroom()
        return convert(state.pop(prefix + name), dtype)
    w['embed'] = take('embed_tokens.weight')
    w['norm'] = take('norm.weight', torch.float32) + 1
    w['head'] = w['embed'] if config['tie_word_embeddings'] else convert(state.pop('lm_head.weight'), torch.float16)
    if config['tie_word_embeddings']:state.pop('lm_head.weight',None)
    for i, kind in enumerate(config['layer_types']):
        layer = f'layers.{i}.'
        w[f'{i}.norm1'] = take(layer+'input_layernorm.weight', torch.float32) + 1
        w[f'{i}.norm2'] = take(layer+'post_attention_layernorm.weight', torch.float32) + 1
        w[f'{i}.gate_up'] = concatenate([take(layer+f'mlp.{name}_proj.weight') for name in ('gate', 'up')])
        w[f'{i}.down'] = take(layer+'mlp.down_proj.weight')
        if kind == 'full_attention':
            w[f'{i}.qkv'] = concatenate([take(layer+f'self_attn.{name}_proj.weight') for name in ('q', 'k', 'v')])
            w[f'{i}.o'] = take(layer+'self_attn.o_proj.weight')
            for name in ('q', 'k'):
                w[f'{i}.{name}_norm'] = take(layer+f'self_attn.{name}_norm.weight', torch.float32) + 1
        else:
            for name in ('qkv', 'z', 'b', 'a'):
                w[f'{i}.{name}'] = take(layer+f'linear_attn.in_proj_{name}.weight')
            for dest, name in (('conv', 'conv1d.weight'), ('out', 'out_proj.weight')):
                w[f'{i}.{dest}'] = take(layer+'linear_attn.'+name)
            for dest, name in (('dt', 'dt_bias'), ('A', 'A_log'), ('gated_norm', 'norm.weight')):
                w[f'{i}.{dest}'] = take(layer+'linear_attn.'+name, torch.float32)
            # The runtime exponentiates A_log per token; a value above exp's
            # overflow threshold yields an infinite decay that silently zeroes
            # the linear-attention state every step. Real loads only: offline
            # meta-tensor planning has no data to validate.
            if not w[f'{i}.A'].is_meta and not torch.isfinite(w[f'{i}.A'].exp()).all():
                raise ValueError('nonfinite hybrid decay rates: A_log overflows')
    return w


class HybridExecutor:
    # Reuse the proven metadata staging and admission checks, with independent state.
    decode_batch = BatchExecutor.decode_batch
    supports_prefix_cache = False

    @property
    def supports_chunked_prefill(self):
        from .device import selected_accelerator
        accelerator = selected_accelerator()
        # Construction has already enforced the Q4 device/precision policy.
        admitted_a30_q4 = (accelerator == 'cuda' and self.model.quantization == 'q4_0'
                           and self.hybrid_precision == 'compensated')
        if getattr(self, 'offload', None) is not None:
            return True
        if accelerator == 'xpu' and self.fused is None:
            return True
        if accelerator == 'mps' and self.fused is None and self.chunk_scan is None:
            # Apple eager-torch path (#4): the per-chunk Metal/torch scan and
            # the prefix_length attention/convolution state machinery are
            # device-independent; qualified by chunked golden replays on the
            # pinned 2B (see docs/evidence/apple/m4-2b-32k/README.md).
            return True
        return ((accelerator == 'rocm' or admitted_a30_q4) and self.fused is not None
                and self.chunk_scan is None)

    @property
    def requires_full_prefill(self):
        return not self.supports_chunked_prefill

    def supports_full_attention_above(self, layer):
        """True when at least one full-attention layer sits above `layer`."""
        return any(kind == 'full_attention'
                   for kind in self.c['layer_types'][layer + 1:self.layer_end])

    def __init__(self, model, use_graph=True, max_batch=4, kernels=None, _layer_range=None):
        if not 1 <= max_batch <= 16:
            raise ValueError('batch capacity must be between 1 and 16')
        from .layer_stream import enabled as offload_enabled, validate as validate_offload
        self.offload = None
        count = model.config['num_hidden_layers']
        if _layer_range is not None:
            if (not isinstance(_layer_range, tuple) or len(_layer_range) != 2
                    or any(type(i) is not int for i in _layer_range)
                    or not 0 <= _layer_range[0] < _layer_range[1] <= count):
                raise ValueError('invalid hybrid stage range')
            if (use_graph or model.quantization not in ('none', 'q8_0', 'q4_0')
                    or offload_enabled()):
                raise ValueError('hybrid stages require unquantized or packed Q8/Q4 eager text without graphs or offload')
            if model.quantization == 'q4_0' and kernels != 'triton':
                raise ValueError('packed Q4 hybrid stages require Triton kernels (the qualified compensated numeric path)')
        self.layer_start, self.layer_end = _layer_range or (0, count)
        streaming = offload_enabled()
        if streaming:
            validate_offload(model.config, model.quantization, model.max_context, max_batch, use_graph, kernels)
        self.device = selected_device()
        use_graph = use_graph and self.device == "cuda"
        self.model, self.c = model, model.config
        self.context, self.max_batch = model.max_context, max_batch
        self.lengths = [0] * max_batch
        self.length = 0
        self.residual_dtype = torch.float32
        c = self.c
        self.eps = c['rms_norm_eps']
        self.heads, self.kh, self.dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        self.lk, self.lv = c['linear_num_key_heads'], c['linear_num_value_heads']
        self.dk, self.dv = c['linear_key_head_dim'], c['linear_value_head_dim']
        self.conv_width = 2*self.lk*self.dk+self.lv*self.dv
        selected = kernels or os.environ.get('DAMENDAEMON_KERNELS', 'auto')
        if selected not in ('auto', 'triton', 'torch'):
            raise ValueError('kernels must be auto, torch, or triton')
        from .hybrid_precision import select as select_precision
        self.hybrid_precision = select_precision(model, use_graph, max_batch, selected)
        self.compute_dtype = torch.float16 if self.hybrid_precision == 'float16' else torch.float32
        cache_element_bytes = 2 if self.compute_dtype == torch.float16 else 4
        if self.hybrid_precision != 'float16' and self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if self.device == 'xpu':
            if selected not in ('auto', 'torch'):
                raise ValueError('Intel hybrid requires eager torch kernels')
            selected = 'torch'
        if self.device == 'mps':
            if selected == 'triton':
                raise ValueError('Triton kernels require CUDA')
            selected = 'torch'
        self.fused = None
        if selected != 'torch':
            try:
                from . import hybrid_kernels
                self.fused = hybrid_kernels
            except ImportError:
                if selected == 'triton':
                    raise
        self.kernel_backend = 'triton-hybrid' if self.fused else 'torch-hybrid'
        if self.hybrid_precision != 'float16':
            if self.fused is None and selected_accelerator() != 'xpu':
                raise ValueError('experimental hybrid precision requires Triton kernels')
            self.kernel_backend += '+' + self.hybrid_precision
        if model.scales:
            self.kernel_backend += '+' + model.quantization
        self.metal_scan = None
        metal_scan = os.environ.get('DAMENDAEMON_HYBRID_METAL_SCAN','auto')
        if metal_scan not in ('auto','0','1'):
            raise ValueError('DAMENDAEMON_HYBRID_METAL_SCAN must be auto, 0 or 1')
        metal_eligible=False
        if self.device=='mps':
            from .hybrid_metal import eligible
            metal_eligible=eligible(c,self.context,max_batch)
        if metal_scan=='1' and not metal_eligible:
            raise ValueError('Metal hybrid scan requires the qualified Qwen3.5-0.8B geometry, context <=1024 and at most four slots')
        if metal_eligible and metal_scan!='0':
            from .hybrid_metal import scan
            self.metal_scan=scan
            self.kernel_backend+='+metal-scan'
        if model.scales and self.fused is None and self.metal_scan is None:
            if self.device != 'xpu':
                raise ValueError('hybrid packed weights require native GPU or qualified Apple Metal kernels')
        self.chunk_scan = None
        prefill_backend = os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan')
        if prefill_backend not in ('scan', 'fla'):
            raise ValueError('DAMENDAEMON_HYBRID_PREFILL must be scan or fla')
        if prefill_backend == 'fla':
            if not self.fused:
                raise ValueError('FLA prefill requires the CUDA Triton backend')
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            self.chunk_scan = chunk_gated_delta_rule
            self.kernel_backend += '+fla-prefill'
        if self.fused:
            from . import kernels as dense_kernels
            self.dense_fused = dense_kernels
        from .hybrid_layout import hybrid_state_bytes
        owned_config = {**c, 'num_hidden_layers': self.layer_end-self.layer_start,
                        'layer_types': c['layer_types'][self.layer_start:self.layer_end]}
        state_bytes = hybrid_state_bytes(owned_config, self.context, max_batch, cache_element_bytes)
        empty_cache()
        free = available_memory()
        # Intel eager projections store FP32 matrices; host byte counts are
        # insufficient because the source mapping retains FP16 matrices.
        # Q8 weights keep their packed int8 storage with FP16 scales on device.
        if self.device == 'xpu':
            device_weight_bytes = sum(t.numel() * (1 if t.dtype == torch.int8 else 4)
                                      for t in {id(t): t for t in model.weights.values()}.values())
            device_weight_bytes += sum(t.numel() * t.element_size()
                                       for t in {id(t): t for t in model.scales.values()}.values())
        else:
            device_weight_bytes = model.bytes
        layout = None
        if streaming:
            from .layer_stream import plan
            layout = plan(model.weights, c['num_hidden_layers'])
            device_weight_bytes = layout.peak_bytes
        # Eager output logits explicitly promote the full output head to FP32.
        # Streaming must budget that temporary in addition to resident weights.
        self.output_head_workspace_bytes = (model.weights['head'].numel()*4
            if streaming and model.weights['head'].dtype != torch.float32 else 0)
        self.stage_activation_workspace_bytes = 0
        # Eager XPU Q8 dequantizes one projection at a time; budget the largest
        # transient FP32 matrix instead of a resident dequantized copy.
        packed_workspace_bytes = 0
        if self.device == 'xpu' and model.scales:
            packed_workspace_bytes = max(t.numel() for t in model.scales.values()) * 32 * 4
        self.required_memory_bytes = (device_weight_bytes + state_bytes + 512*1024**2
                                      + self.output_head_workspace_bytes + self.stage_activation_workspace_bytes
                                      + packed_workspace_bytes)
        if _layer_range is not None:
            from .hybrid_layout import stage_budget
            budget = stage_budget(model.weights, c, self.context, max_batch, cache_element_bytes,
                                  self.layer_start, self.layer_end, fp32_weights=self.device == 'xpu',
                                  scales=model.scales)
            self.required_memory_bytes = budget['required_memory_bytes']
            self.output_head_workspace_bytes = budget['output_head_workspace_bytes']
            self.stage_activation_workspace_bytes = budget['activation_workspace_bytes']
        if self.required_memory_bytes > free:
            raise MemoryError('insufficient GPU memory for hybrid weights, state, and workspace')
        if streaming:
            from .layer_stream import LayerWeightStore
            self.offload = LayerWeightStore(model.weights, layout,
                lambda tensor, device=self.device: tensor.to(device), synchronize, empty_cache)
            self.w = self.offload.resident
        else:
            copies = {}
            self.w = {}
            for key, value in model.weights.items():
                if id(value) not in copies:
                    copies[id(value)] = value.to(device=self.device,
                        dtype=torch.float32 if self.device == 'xpu' and value.dtype != torch.int8 else value.dtype)
                self.w[key] = copies[id(value)]
        self.linear = F.linear
        if model.scales:
            from .quantization import Linear
            projection = projection_q4 = None
            if self.hybrid_precision == 'compensated':
                from .compensated_q8 import projection, projection_q4
            self.linear = Linear(self.w, model.scales, self.device, F.linear,
                                 q8_projection=projection, q4_projection=projection_q4)
        self.kv, self.conv, self.state = {}, {}, {}
        # Q8_0 KV storage cuts the FP32 cache to roughly a quarter of its bytes
        # (plus 1/32-scale FP16 overhead); values are dequantized to compute
        # dtype on read. Opt-in via DAMENDAEMON_HYBRID_KV=q8_0.
        self.kv_quant = os.environ.get('DAMENDAEMON_HYBRID_KV', 'none')
        if self.kv_quant not in ('none', 'q8_0'):
            raise ValueError('DAMENDAEMON_HYBRID_KV must be none or q8_0')
        # TokenGate opt-in (#641): default off; production paths are
        # byte-identical with the flag unset (the gated surface lives in
        # gated_prefill/gated_prefill_chunk and the decode hole mask).
        self.gate = None
        self.gate_valid = None
        self.gate_fold = None
        self.gate_nvalid = None
        self._gate_cache = {}
        if os.environ.get('DAMENDAEMON_TOKEN_GATE', '0') == '1':
            from . import token_gate as token_gate_module
            if _layer_range is not None:
                raise ValueError('token gating is a full-executor surface, not a hybrid stage')
            if use_graph:
                raise ValueError('token gating requires eager decode (CUDA graphs would bake per-slot hole masks)')
            if self.kv_quant != 'none':
                raise ValueError('token gating is not yet qualified with quantized KV')
            if not self.supports_full_attention_above(token_gate_module.configured_layer()):
                raise ValueError('token gate needs at least one full-attention layer above the gate layer')
            layer = token_gate_module.configured_layer()
            keep = token_gate_module.configured_keep()
            weights_path = os.environ.get('DAMENDAEMON_TOKEN_GATE_WEIGHTS')
            if weights_path:
                self.gate = TokenGate.from_upstream(weights_path, c, layer, keep)
            else:
                self.gate = TokenGate.untrained(self, layer, keep)
            self.gate.to_device(self.device)
            self.gate_valid = {i: torch.zeros(max_batch, self.context, dtype=torch.bool, device=self.device)
                               for i in self.gate.bias_layers}
            # Per-slot additive decode bias per gated full-attention layer:
            # the upstream decode fold (head-mean key bias, constant through
            # decode) plus a hole mask over never-written KV positions.
            self.gate_fold = {i: torch.zeros(max_batch, self.context, dtype=torch.float32, device=self.device)
                              for i in self.gate.bias_layers}
            self.gate_nvalid = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        # Fused kernels and quantized KV coexist: only dense_fused.attention
        # reads the KV cache, and the quantized attention branch below bypasses
        # it in favor of eager scaled attention. Fused linear scan/conv and
        # fused norms operate on conv/state/weights, never on kv.
        for i in range(self.layer_start, self.layer_end):
            kind = c['layer_types'][i]
            if kind == 'full_attention':
                if self.kv_quant == 'q8_0':
                    from .kv_quant import allocate_kv
                    self.kv[i] = allocate_kv(max_batch, self.kh, self.context, self.dim, self.device)
                else:
                    self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=self.compute_dtype, device=self.device)
            else:
                self.conv[i] = torch.zeros((max_batch, self.conv_width, c['linear_conv_kernel_dim']), dtype=self.compute_dtype, device=self.device)
                self.state[i] = torch.zeros((max_batch, self.lv, self.dk, self.dv), dtype=torch.float32, device=self.device)
        # Gate-3 accounting (#535): resident KV bytes per mode, reported in
        # /v1/models/status. The quantized pair counts int8 data plus FP16
        # scales; the plain cache counts its compute-dtype bytes.
        self.kv_mode = self.kv_quant
        total = 0
        for cache in self.kv.values():
            if isinstance(cache, tuple):
                data, scales = cache
                total += data.numel() + scales.numel()*scales.element_size()
            else:
                total += cache.numel()*cache.element_size()
        self.kv_cache_bytes = total
        dim = c['rotary_dim']
        inv = c['rope_theta'] ** (-torch.arange(0, dim, 2, device=self.device, dtype=torch.float32)/dim)
        angles = torch.outer(torch.arange(self.context, device=self.device), inv)
        angles = torch.cat((angles, angles), -1)
        self.cos, self.sin = angles.cos(), angles.sin()
        self.arange = torch.arange(self.context, device=self.device)
        if self.fused and _layer_range is None:
            # A layer stage never runs the full backbone: its forward_embeddings
            # is deliberately blocked (the pipeline coordinator owns layer
            # execution and cursor commit), so the fused warmup probe below is
            # a full-model-only step. Stage kernels warm on the pipeline's
            # first forward_layers, and the toolchain-fallback branch here is
            # an unquantized-FP16 full-model concern.
            from .cuda_warmup import warmup_stream
            probe = torch.zeros(1, dtype=torch.long, device=self.device)
            with warmup_stream(), torch.inference_mode():
                try:
                    self.forward_batch(probe, probe, probe)
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError) as error:
                    if selected != 'auto' or self.hybrid_precision != 'float16':
                        raise
                    if self.chunk_scan is not None:
                        raise
                    self.fused = None
                    self.kernel_backend = 'torch-hybrid'
                    print(f'damendaemon: hybrid kernel toolchain unavailable ({type(error).__name__}); using torch kernels', file=sys.stderr)
            torch.cuda.synchronize()
        self.batch_graphs = {}
        if use_graph:
            from .cuda_warmup import warmup_stream
            for size in range(1, max_batch+1):
                metadata = torch.zeros((3, size), dtype=torch.long, device=self.device)
                ids, positions, slots = metadata.unbind(0)
                slots.copy_(torch.arange(size, device=self.device))
                staging = torch.empty((3, size), dtype=torch.long, pin_memory=True)
                values = (ctypes.c_int64 * (3*size)).from_address(staging.data_ptr())
                copied = torch.cuda.Event()
                with warmup_stream(), torch.inference_mode():
                    for _ in range(3):
                        self.forward_batch(ids, positions, slots)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph):
                    logits = self.forward_batch(ids, positions, slots)
                    next_ids = logits.argmax(-1)
                self.batch_graphs[size] = (metadata, staging, values, copied, graph, logits, next_ids)
        # Capturing recurrent updates changes state. No captured warmup is user history.
        for group in (*self.conv.values(), *self.state.values(), *self.kv.values()):
            for tensor in (group if isinstance(group, tuple) else (group,)):
                tensor.zero_()

    def snapshot_prefix(self, slot, count):
        """Snapshot the KV prefix [0:count] of one slot for the prefix cache."""
        snapshots = []
        for layer in self.kv.values():
            if isinstance(layer, tuple):
                data, scales = layer
                snapshots.append((data[:, slot, :, :count, :].clone(),
                                  scales[:, slot, :, :count, :].clone()))
            else:
                snapshots.append(layer[:, slot, :, :count, :].clone())
        return snapshots

    def restore_prefix(self, slot, reused, cache):
        """Restore a KV prefix snapshot captured by snapshot_prefix."""
        for layer, saved in zip(self.kv.values(), cache):
            if isinstance(layer, tuple):
                data, scales = layer
                saved_data, saved_scales = saved
                data[:, slot, :, :reused, :].copy_(saved_data[:, :, :, :reused, :])
                scales[:, slot, :, :reused, :].copy_(saved_scales[:, :, :, :reused, :])
            else:
                layer[:, slot, :, :reused, :].copy_(saved[:, :, :, :reused, :])

    def prefix_state_bytes(self, count):
        """Resident bytes a one-slot KV prefix of count tokens occupies."""
        total = 0
        for layer in self.kv.values():
            if isinstance(layer, tuple):
                data, scales = layer
                total += 2*data.shape[2]*data.shape[-1]*count*data.element_size()
                total += 2*scales.shape[2]*scales.shape[-1]*count*scales.element_size()
            else:
                total += 2*layer.shape[2]*layer.shape[-1]*count*layer.element_size()
        return total

    @property
    def failed(self):
        return self.offload is not None and self.offload.failed

    def close(self):
        if self.offload is not None:
            self.offload.close()

    def norm(self, x, weight, dtype=None):
        if dtype is None:
            dtype = self.compute_dtype
        if self.fused:
            return self.fused.norm(x, weight, self.eps, dtype)
        x = x.float()
        variance = x.square().mean(-1, keepdim=True)
        # Finite inputs can overflow here and otherwise silently normalize to zero.
        if not torch.isfinite(variance).all():raise RuntimeError('nonfinite hybrid normalization variance')
        result = (x * torch.rsqrt(variance+self.eps) * weight).to(dtype)
        if not torch.isfinite(result.float()).all():raise RuntimeError('nonfinite hybrid normalization output')
        return result

    def rotate(self, x, positions):
        dim = self.c['rotary_dim']
        rotated, rest = x[..., :dim], x[..., dim:]
        left, right = rotated.chunk(2, -1)
        cos, sin = self.cos[positions].unsqueeze(-2), self.sin[positions].unsqueeze(-2)
        rotated = rotated*cos + torch.cat((-right, left), -1)*sin
        return torch.cat((rotated, rest), -1).to(self.compute_dtype)

    def linear_attention(self, x, i, slots, prefill, prefix_length=0):
        w = self.w
        n, length, _ = x.shape
        qkv = self.linear(x, w[f'{i}.qkv'])
        z = self.linear(x, w[f'{i}.z']).view(n, length, self.lv, self.dv)
        beta = self.linear(x, w[f'{i}.b']).float().sigmoid()
        decay = -w[f'{i}.A'].exp()*F.softplus(self.linear(x, w[f'{i}.a']).float()+w[f'{i}.dt'])
        if prefill:
            history = qkv.transpose(1, 2).contiguous()
            width = self.c['linear_conv_kernel_dim']
            if prefix_length and width > 1:
                history = torch.cat((self.conv[i].index_select(0, slots)[..., -(width-1):], history), -1)
            self.conv[i][slots] = F.pad(history, (max(0, width-length), 0))[..., -width:]
            convolved = F.conv1d(history.float(), w[f'{i}.conv'].float(),
                padding=0 if prefix_length else width-1, groups=self.conv_width)[..., :length]
            qkv = F.silu(convolved).transpose(1, 2)
        elif self.fused:
            qkv = self.fused.conv_step(qkv[:, 0].contiguous(), w[f'{i}.conv'], self.conv[i], slots).unsqueeze(1)
        else:
            history = torch.cat((self.conv[i].index_select(0, slots)[..., 1:], qkv.transpose(1, 2)), -1)
            self.conv[i][slots] = history
            qkv = F.silu((history.float()*w[f'{i}.conv'].squeeze(1).float()).sum(-1)).unsqueeze(1)
        q, k, v = qkv.split((self.lk*self.dk, self.lk*self.dk, self.lv*self.dv), -1)
        q, k = [a.reshape(n, length, self.lk, self.dk) for a in (q, k)]
        v = v.reshape(n, length, self.lv, self.dv)
        if prefill and length >= 64 and self.chunk_scan is not None:
            repeat = self.lv // self.lk
            out, last_state = self.chunk_scan(q.repeat_interleave(repeat, 2).contiguous(),
                k.repeat_interleave(repeat, 2).contiguous(), v.contiguous(),
                g=decay.contiguous(), beta=beta.contiguous(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            self.state[i][slots] = last_state
        elif self.fused:
            out = self.fused.scan(q, k, v, decay, beta, self.state[i], slots,
                                  reset=prefill and not prefix_length)
        else:
            q, k = [_l2_normalize(a) for a in (q, k)]
            q, k = [a.repeat_interleave(self.lv//self.lk, dim=2) for a in (q, k)]
            q = q * (self.dk**-.5)
            if self.metal_scan is not None:
                # Reset only on the FIRST chunk of a prompt: continuation
                # chunks (prefix_length set) must accumulate the recurrent
                # state, exactly like the torch fallback below. Passing raw
                # `prefill` wiped the state on every chunk (#4 bug found by
                # the chunked 4K golden replay: 13/84 slot-0 checks failed
                # with ~0.2 relative RMS, greedy-preserving).
                out=self.metal_scan(q,k,v,decay,beta,self.state[i],slots,
                                    prefill and not prefix_length)
            else:
                state = (torch.zeros_like(self.state[i].index_select(0, slots))
                         if prefill and not prefix_length else self.state[i].index_select(0, slots))
                outputs = []
                for t in range(length):
                    state = state * decay[:, t].exp()[..., None, None]
                    delta = (v[:, t].float()-(state*k[:, t, :, :, None]).sum(-2))*beta[:, t, :, None]
                    state = state + k[:, t, :, :, None]*delta.unsqueeze(-2)
                    outputs.append((state*q[:, t, :, :, None]).sum(-2))
                self.state[i][slots] = state
                out = torch.stack(outputs, 1)
        if self.fused:
            out = self.fused.norm(out, w[f'{i}.gated_norm'], self.eps, self.compute_dtype, gate=z)
        else:
            out = self.norm(out, w[f'{i}.gated_norm'], torch.float32) * F.silu(z.float())
        return self.linear(out.reshape(n, length, -1).to(self.compute_dtype), w[f'{i}.out'])

    def attention(self, x, i, positions, slots, prefill, prefix_length=0):
        n, length, _ = x.shape
        w = self.w
        q, k, v = self.linear(x, w[f'{i}.qkv']).split((2*self.heads*self.dim, self.kh*self.dim, self.kh*self.dim), -1)
        q, gate = q.view(n, length, self.heads, 2*self.dim).chunk(2, -1)
        k, v = [a.view(n, length, self.kh, self.dim) for a in (k, v)]
        q = self.rotate(self.norm(q, w[f'{i}.q_norm'], torch.float32), positions)
        k = self.rotate(self.norm(k, w[f'{i}.k_norm'], torch.float32), positions)
        if self.kv_quant == 'q8_0':
            from .kv_quant import store_kv, read_kv
            # store/read use (batch, heads, tokens, dim) and per-row (batch,
            # tokens) positions — the cache's own layout. Decode positions are
            # already (batch, 1) from forward_batch.
            store_kv(self.kv[i], slots, positions, k.transpose(1, 2), v.transpose(1, 2))
            # Decode reads each slot's full history (absolute positions); the
            # attention mask excludes unfilled positions of shorter slots.
            end = prefix_length + length if prefill else int(positions.max()) + 1
            # read_kv returns slot-indexed (slots, kv_heads, end, dim) in the
            # compute dtype — the same outer-slot layout the FP32 cache paths
            # produce.
            ck, cv = read_kv(self.kv[i], slots, end, self.compute_dtype)
        else:
            ck, cv = self.kv[i].unbind(0)
            ck[slots[:, None], :, positions, :] = k
            cv[slots[:, None], :, positions, :] = v
        from .grouped_attention import use_grouped_prefill
        from .device import selected_accelerator
        end = prefix_length + length
        if self.kv_quant == 'q8_0':
            # Slot-indexed views of the populated prefix in the compute dtype.
            # The quantized path feeds eager attention only; the fused dense
            # decode kernel reads the FP32 cache layout and is bypassed here
            # (fused linear scan/conv and fused norms are KV-independent).
            if use_grouped_prefill(selected_accelerator(), self.hybrid_precision, prefill, length, end):
                from .grouped_attention import attention as grouped_attention
                out = grouped_attention(q.transpose(1, 2), ck, cv)
            elif prefill and prefix_length:
                from torch.nn.attention.bias import causal_lower_right
                mask = (self.arange[:end][None, :] <= positions[0, :, None]
                        if selected_accelerator() == 'xpu' or self.offload is not None
                        else causal_lower_right(length, end))
                out = scaled_attention(q.transpose(1, 2), ck, cv,
                    attn_mask=mask, enable_gqa=True)
            elif prefill:
                out = scaled_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                             is_causal=True, enable_gqa=True)
            else:
                mask = self.arange.view(1, 1, 1, -1)[:, :, :, :end] <= positions.view(n, 1, 1, 1)
                out = scaled_attention(q.transpose(1, 2), ck, cv,
                                                             attn_mask=mask, enable_gqa=True)
        elif use_grouped_prefill(selected_accelerator(), self.hybrid_precision, prefill, length, end):
            from .grouped_attention import attention as grouped_attention
            out = grouped_attention(q.transpose(1, 2),
                ck[:, :, :end].index_select(0, slots), cv[:, :, :end].index_select(0, slots))
        elif prefill and prefix_length:
            from torch.nn.attention.bias import causal_lower_right
            end = prefix_length + length
            mask = (self.arange[:end][None, :] <= positions[0, :, None]
                    if selected_accelerator() == 'xpu' or self.offload is not None
                    else causal_lower_right(length, end))
            out = scaled_attention(q.transpose(1, 2),
                ck[:, :, :end].index_select(0, slots), cv[:, :, :end].index_select(0, slots),
                attn_mask=mask, enable_gqa=True)
        elif prefill:
            out = scaled_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                 is_causal=True, enable_gqa=True)
        elif self.fused and (self.gate is None or i not in self.gate_valid):
            out = self.dense_fused.attention(q.transpose(1, 2), ck, cv, positions[:, 0].contiguous(), slots)
        else:
            mask = self.arange.view(1, 1, 1, -1) <= positions.view(n, 1, 1, 1)
            if self.gate is not None and i in self.gate_valid:
                # Gated decode (#641): never-written KV positions (pruned
                # tokens) are holes; the query carries its own temperature
                # (r · softplus(sigma)); the per-key bias enters as the
                # upstream head-mean fold. Eager only — the fused dense
                # decode kernel has no mask support.
                valid = self.gate_valid[i][slots].view(n, 1, 1, -1)
                mask = mask & valid
                fold = self.gate_fold[i][slots][:, None, None, :].to(q.dtype)
                row = self.gate.bias_layers.index(i)
                # _gate_decode_step already counted this token: m is the
                # current valid count (kept + generated so far, inclusive).
                m = self.gate_nvalid[slots].clamp(min=2).float()
                p = (positions[:, 0] + 1).clamp(min=2).float()
                ratio = (m.log() / p.log()).clamp(max=1.0)
                # r and sigma compute in fp32; the product returns to the
                # backbone's decode dtype so SDPA sees q/k/v uniformly.
                scale = (ratio.view(n, 1, 1, 1)
                         * self.gate.temperature(row).view(1, 1, -1, 1).float()).to(q.dtype)
                q = q * scale.view(n, 1, -1, 1)
                additive = torch.where(mask, fold.expand(n, 1, 1, mask.shape[-1]),
                                       torch.full_like(fold, float('-inf')).expand(n, 1, 1, mask.shape[-1]))
                out = scaled_attention(q.transpose(1, 2), ck.index_select(0, slots), cv.index_select(0, slots),
                                                     attn_mask=additive, enable_gqa=True)
            else:
                out = scaled_attention(q.transpose(1, 2), ck.index_select(0, slots), cv.index_select(0, slots),
                                                     attn_mask=mask, enable_gqa=True)
        out = out.transpose(1, 2).float()*gate.float().sigmoid()
        return self.linear(out.reshape(n, length, -1).to(self.compute_dtype), w[f'{i}.o'])

    @torch.inference_mode()
    def forward(self, ids, positions, slots, prefill, prefix_length=0, return_logits=True):
        if self.gate is not None and not prefill:
            self._gate_decode_step(slots, positions)
        return self.forward_embeddings(self.embedding(ids),positions,slots,prefill,
                                       prefix_length, return_logits)

    def _gate_decode_step(self, slots, positions):
        """A decoded token is never pruned: mark its position valid in every
        gated layer (hole mask admits it) and count it for the next step's
        r-ratio, before any attention consumes the mask."""
        for row, slot in enumerate(slots.tolist()):
            position = int(positions[row, 0])
            if not 0 <= position < self.context:
                raise ValueError('invalid decode position for the token gate')
            for i in self.gate_valid:
                self.gate_valid[i][slot, position] = True
            self.gate_nvalid[slot] += 1

    def embedding(self, ids):
        return F.embedding(ids, self.w['embed']).float()

    def forward_embeddings(self, x, positions, slots, prefill, prefix_length=0, return_logits=True):
        """Shared text backbone after token or image/text embedding assembly."""
        x = self.forward_layers(x, positions, slots, prefill, prefix_length)
        return self.output_logits(x[:, -1]) if return_logits else None

    def forward_layers(self, x, positions, slots, prefill, prefix_length=0,
                       layer_start=None, layer_end=None):
        """Execute a contiguous range using global layer indices and positions.

        Returns every hidden position. The caller owns embedding, final-head
        selection and request cursor commit across all participating ranges.
        """
        if layer_start is None:
            layer_start = self.layer_start
        if layer_end is None:
            layer_end = self.layer_end
        if (type(layer_start) is not int or type(layer_end) is not int
                or not self.layer_start <= layer_start < layer_end <= self.layer_end):
            raise ValueError('invalid hybrid layer range')
        for i in range(layer_start, layer_end):
            x = self._run_layer(x, i, positions, slots, prefill, prefix_length)
        return x

    def _run_layer(self, x, i, positions, slots, prefill, prefix_length):
        """One full decoder layer: mixer half then MLP half."""
        from .device import selected_accelerator
        if self.offload is not None:
            self.offload.activate(i)
        if selected_accelerator() == 'rocm':
            from .rocm_policy import require_host_headroom
            require_host_headroom()
        h = self.norm(x, self.w[f'{i}.norm1'])
        x = x + (self.linear_attention(h, i, slots, prefill, prefix_length) if self.c['layer_types'][i] == 'linear_attention'
                 else self.attention(h, i, positions, slots, prefill, prefix_length))
        return self._mlp_half(x, i)

    def _mlp_half(self, x, i):
        """The MLP sublayer of one decoder layer: a + MLP(LN2(a))."""
        gate_up = self.linear(self.norm(x, self.w[f'{i}.norm2']), self.w[f'{i}.gate_up'])
        if self.fused:
            activated = self.dense_fused.swiglu(gate_up.reshape(-1, gate_up.shape[-1]), True)
            activated = activated.view(*gate_up.shape[:-1], -1)
        else:
            gate, up = gate_up.chunk(2, -1)
            activated = (F.silu(gate.float())*up.float()).to(self.compute_dtype)
        return x + self.linear(activated, self.w[f'{i}.down'])

    def output_logits(self, hidden):
        normalized = self.norm(hidden, self.w['norm'])
        if self.w['head'].dtype in (torch.int8, torch.uint8):
            return self.linear(normalized.float(), self.w['head'])
        if self.fused:
            return self.fused.output_head(normalized, self.w['head'])
        return F.linear(normalized.float(), self.w['head'].float())

    def forward_batch(self, ids, positions, slots):
        return self.forward(ids[:, None], positions[:, None], slots, False)

    @torch.inference_mode()
    def prefill(self, tokens, slot=0):
        if not tokens or len(tokens) > self.context or not 0 <= slot < self.max_batch:
            raise ValueError('invalid prompt or cache slot')
        if set(tokens) & set(self.c.get('multimodal_token_ids', [])):
            raise ValueError('multimodal input is not supported')
        result = self.forward(torch.tensor([tokens], device=self.device),
            torch.arange(len(tokens), device=self.device)[None], torch.tensor([slot], device=self.device), True)
        self.lengths[slot] = len(tokens)
        self.length = self.lengths[0]
        self._gate_mark_ungated(slot, len(tokens))
        return result

    def _gate_mark_ungated(self, slot, count):
        """An ungated prefill under a gate-enabled executor fills every hole:
        all positions valid, no decode bias, so decode masks are uniform."""
        if self.gate is None:
            return
        for i in self.gate_valid:
            self.gate_valid[i][slot, :count] = True
            self.gate_fold[i][slot, :count] = 0.0
        self.gate_nvalid[slot] = count

    def _gate_reset_slot(self, slot):
        if self.gate is None:
            return
        for i in self.gate_valid:
            self.gate_valid[i][slot].fill_(False)
            self.gate_fold[i][slot].fill_(0.0)
        self.gate_nvalid[slot] = 0
        self._gate_cache.pop(slot, None)

    def forward_boundary(self, x, positions, slots, prefill, prefix_length=0):
        """Phase 1 of the gated path: every layer below the gate layer in
        full, then the gate layer's MIXER half only — returning the phase
        boundary a_j (post-attention residual, MLP half pending) for every
        position. Layers at or below the gate keep their full phase-1 state
        (KV/conv/recurrent) over every token; nothing above runs."""
        gate_layer = self.gate.gate_layer
        for i in range(self.layer_start, gate_layer):
            x = self._run_layer(x, i, positions, slots, prefill, prefix_length)
        from .device import selected_accelerator
        if self.offload is not None:
            self.offload.activate(gate_layer)
        if selected_accelerator() == 'rocm':
            from .rocm_policy import require_host_headroom
            require_host_headroom()
        h = self.norm(x, self.w[f'{gate_layer}.norm1'])
        x = x + (self.linear_attention(h, gate_layer, slots, prefill, prefix_length)
                 if self.c['layer_types'][gate_layer] == 'linear_attention'
                 else self.attention(h, gate_layer, positions, slots, prefill, prefix_length))
        return x

    def _gated_attention(self, x, i, kept_positions, slots, ratio, key_bias):
        """Phase-2 full-attention (#641): KV written only at kept positions,
        logits tempered per query and biased per key:

            softmax((r_q · softplus(sigma)) · q·k/√d + b_k) over kept keys.

        ``ratio`` is r_q per kept position; ``key_bias`` is b_k per kept
        position for this layer's gate row (zero on non-article keys).
        Causality runs over the kept sequence, which preserves original
        order (kept positions are sorted), so an in-pass causal mask is
        exact.
        """
        n, length, _ = x.shape
        w = self.w
        q, k, v = self.linear(x, w[f'{i}.qkv']).split((2*self.heads*self.dim, self.kh*self.dim, self.kh*self.dim), -1)
        q, gate = q.view(n, length, self.heads, 2*self.dim).chunk(2, -1)
        k, v = [a.view(n, length, self.kh, self.dim) for a in (k, v)]
        positions = kept_positions.view(1, -1)
        q = self.rotate(self.norm(q, w[f'{i}.q_norm'], torch.float32), positions)
        k = self.rotate(self.norm(k, w[f'{i}.k_norm'], torch.float32), positions).float()
        q = q.float()  # gated attention computes in fp32 regardless of the
        # backbone's compute dtype (property suite reference); the cache
        # write below casts back to the cache dtype, matching plain decode.
        ck, cv = self.kv[i].unbind(0)
        ck[slots[:, None], :, positions[0], :] = k.to(ck.dtype)
        cv[slots[:, None], :, positions[0], :] = v.to(cv.dtype)
        row = self.gate.bias_layers.index(i)
        temperature = (ratio.view(1, -1, 1, 1).float()
                       * self.gate.temperature(row).view(1, 1, -1, 1).float())
        q = q.float() * temperature
        # (n, heads, q_len, kv_len): GQA repeats key/value heads to query
        # heads — each query head attends with its own kv head, exactly as
        # SDPA's enable_gqa path does.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.heads // self.kh
        if repeat > 1:
            k = k.repeat_interleave(repeat, 1)
            v = v.repeat_interleave(repeat, 1)
        logits = q @ k.transpose(-1, -2) * (self.dim ** -.5)
        # Per-(query head, key) additive bias — b_k[l,h], zero on
        # non-article keys. (t() + unsqueeze: no .view on a sliced tensor.)
        logits = logits + key_bias.t()[None, :, None, :].float()
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        logits = logits.masked_fill(~causal, float('-inf'))
        weights = torch.softmax(logits, -1)
        out = weights @ v.float()
        out = out.transpose(1, 2) * gate.float().sigmoid()
        return self.linear(out.reshape(n, length, -1).to(self.compute_dtype), w[f'{i}.o'])

    def _gated_phase2(self, boundary, kept_positions, bias, slots):
        """Run the kept sequence from the layer-L phase boundary through the
        rest of the backbone: the gate layer's MLP half on kept tokens, then
        every layer above — full-attention layers through _gated_attention
        (KV at kept positions only), DeltaNet layers over the kept sequence
        (no bias: no per-token KV to bias, per upstream §GDN placement).

        ``boundary`` covers the whole prompt; ``kept_positions`` selects.
        ``bias`` is the (rows, heads) per-key bias already gathered to the
        kept sequence (zero rows for non-article keys).
        """
        gate_layer = self.gate.gate_layer
        kept = kept_positions.view(1, -1)
        x = boundary.index_select(1, kept_positions)
        x = self._mlp_half(x, gate_layer)
        ratio = TokenGate.kept_ratio(kept_positions)
        for i in range(gate_layer + 1, self.layer_end):
            if self.c['layer_types'][i] == 'linear_attention':
                h = self.norm(x, self.w[f'{i}.norm1'])
                x = x + self.linear_attention(h, i, slots, True, 0)
            else:
                row = self.gate.bias_layers.index(i)
                h = self.norm(x, self.w[f'{i}.norm1'])
                x = x + self._gated_attention(h, i, kept_positions, slots, ratio,
                                              bias[:, row])
        return x

    @torch.inference_mode()
    def gated_prefill(self, tokens, slot, article_start, article_end, keep=None):
        """Gated whole-prompt prefill (#641): the prunable span
        [article_start, article_end) is scored at the gate layer's phase
        boundary; layers above the gate process only the kept set (top-K
        article tokens + every token outside the span). An all-zero score
        vector (untrained gate) falls back to the standard prefill path,
        byte-exactly, at any configured K."""
        if (self.gate is None or not tokens or len(tokens) > self.context
                or not 0 <= slot < self.max_batch or self.lengths[slot] != 0
                or not 0 <= article_start < article_end <= len(tokens)
                or set(tokens) & set(self.c.get('multimodal_token_ids', []))):
            raise ValueError('invalid gated prompt, span, or cache slot')
        try:
            ids = torch.tensor([tokens], device=self.device)
            positions = torch.arange(len(tokens), device=self.device)[None]
            slots = torch.tensor([slot], device=self.device)
            boundary = self.forward_boundary(self.embedding(ids), positions, slots, True)
            scores = self.gate.score_tokens(boundary, self.w[f'{self.gate.gate_layer}.norm2'], self.eps)
            article = torch.zeros(len(tokens), dtype=torch.bool, device=self.device)
            article[article_start:article_end] = True
            flat_article = scores[0][article]
            if not flat_article.any():
                # Untrained/inactive gate: standard path, byte-exact, any K.
                result = self.forward(ids, positions, slots, True)
                self.lengths[slot] = len(tokens)
                self.length = self.lengths[0]
                self._gate_mark_ungated(slot, len(tokens))
                return result
            budget = self.gate.keep if keep is None else keep
            kept_article = self.gate.select(scores, article, budget)
            kept = torch.sort(torch.cat((torch.arange(len(tokens), device=self.device)[~article],
                                         kept_article))).values
            bias = self.gate.bias(scores, article)[0][kept]
            hidden = self._gated_phase2(boundary, kept, bias, slots)
            logits = self.output_logits(hidden[:, -1])
        except Exception:
            # A failed layer may already have changed recurrent state. Only a
            # new prompt at offset zero can safely reuse this slot afterwards.
            self.lengths[slot] = 0
            self.length = self.lengths[0]
            self._gate_reset_slot(slot)
            raise
        self.lengths[slot] = len(tokens)
        self.length = self.lengths[0]
        for i in self.gate_valid:
            self.gate_valid[i][slot, kept] = True
            self.gate_fold[i][slot, kept] = self.gate.decode_fold(bias, kept)[:, self.gate.bias_layers.index(i)]
        self.gate_nvalid[slot] = kept.numel()
        return logits

    @torch.inference_mode()
    def gated_prefill_chunk(self, tokens, slot, start, final, article_start, article_end, keep=None):
        """Gated chunked prefill (#641): non-final chunks run only through
        the gate layer's boundary, caching per-token boundary rows; the final
        chunk assembles the whole-prompt boundary, makes ONE global top-K
        selection over every cached article score, and runs the deferred
        phase-2 over the whole kept set — so chunked gating equals whole
        gating by construction (the #585 chunk-equivalence pattern)."""
        if (self.gate is None or not tokens or start + len(tokens) > self.context
                or not 0 <= article_start < article_end
                or (final and not article_start < article_end <= start + len(tokens))
                or article_end > self.context
                or (start and self.lengths[slot] != start)
                or not 0 <= slot < self.max_batch or type(final) is not bool
                or set(tokens) & set(self.c.get('multimodal_token_ids', []))):
            raise ValueError('invalid partial gated prompt or cache state')
        try:
            ids = torch.tensor([tokens], device=self.device)
            positions = torch.arange(start, start + len(tokens), device=self.device)[None]
            slots = torch.tensor([slot], device=self.device)
            boundary = self.forward_boundary(self.embedding(ids), positions, slots, True, start)
            cache = self._gate_cache.setdefault(slot, {'rows': [], 'tokens': [], 'count': 0})
            if start == 0:
                cache['rows'] = []
                cache['tokens'] = []
                cache['count'] = 0
            cache['rows'].append(boundary[0].float())
            cache['tokens'].extend(tokens)
            cache['count'] += len(tokens)
            if not final:
                self.lengths[slot] = start + len(tokens)
                self.length = self.lengths[0]
                return None
            whole = torch.cat(cache['rows'], 0)[None]
            total = cache['count']
            tokens_all = cache['tokens']
            scores = self.gate.score_tokens(whole, self.w[f'{self.gate.gate_layer}.norm2'], self.eps)
            article = torch.zeros(total, dtype=torch.bool, device=self.device)
            article[article_start:article_end] = True
            flat_article = scores[0][article]
            if not flat_article.any():
                result = self.forward(torch.tensor([tokens_all], device=self.device),
                    torch.arange(total, device=self.device)[None], slots, True)
                self.lengths[slot] = total
                self.length = self.lengths[0]
                self._gate_mark_ungated(slot, total)
                self._gate_cache.pop(slot, None)
                return result
            budget = self.gate.keep if keep is None else keep
            kept_article = self.gate.select(scores, article, budget)
            kept = torch.sort(torch.cat((torch.arange(total, device=self.device)[~article],
                                         kept_article))).values
            bias = self.gate.bias(scores, article)[0][kept]
            hidden = self._gated_phase2(whole, kept, bias, slots)
            logits = self.output_logits(hidden[:, -1])
        except Exception:
            self.lengths[slot] = 0
            self.length = self.lengths[0]
            self._gate_reset_slot(slot)
            raise
        self.lengths[slot] = total
        self.length = self.lengths[0]
        for i in self.gate_valid:
            self.gate_valid[i][slot, kept] = True
            self.gate_fold[i][slot, kept] = self.gate.decode_fold(bias, kept)[:, self.gate.bias_layers.index(i)]
        self.gate_nvalid[slot] = kept.numel()
        self._gate_cache.pop(slot, None)
        return logits

    def prefill_batch(self, prompts, slots):
        if (not prompts or len(prompts) != len(slots) or len(set(slots)) != len(slots)
                or any(s < 0 or s >= self.max_batch for s in slots)
                or any(not p or len(p) > self.context for p in prompts)
                or any(set(p) & set(self.c.get('multimodal_token_ids', [])) for p in prompts)):
            raise ValueError('invalid prefill batch')
        # Different lengths are independently prefetched. Decode remains continuously batched.
        return torch.cat([self.prefill(p, s) for p, s in zip(prompts, slots)])

    @torch.inference_mode()
    def prefill_chunk(self, tokens, slot, start, final=True):
        if not self.supports_chunked_prefill or self.requires_full_prefill:
            raise ValueError('hybrid chunked prefill requires an admitted scan text path')
        if (type(slot) is not int or not 0 <= slot < self.max_batch
                or type(start) is not int or start < 0 or type(final) is not bool
                or not tokens or start + len(tokens) > self.context
                or (start and self.lengths[slot] != start)
                or set(tokens) & set(self.c.get('multimodal_token_ids', []))):
            raise ValueError('invalid partial hybrid prompt or cache state')
        try:
            logits = self.forward(torch.tensor([tokens], device=self.device),
                torch.arange(start, start + len(tokens), device=self.device)[None],
                torch.tensor([slot], device=self.device), True, start, final)
        except Exception:
            # A failed layer may already have changed recurrent state. Only a
            # new prompt at offset zero can safely reuse this slot afterwards.
            self.lengths[slot] = 0
            self.length = self.lengths[0]
            raise
        self.lengths[slot] = start + len(tokens)
        self.length = self.lengths[0]
        if final:
            self._gate_mark_ungated(slot, start + len(tokens))
        return logits

    def decode(self, token):
        result = self.decode_batch([token], [0])
        self.length = self.lengths[0]
        return result
