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


def load_weights(state, config):
    # Multimodal checkpoints wrap the text backbone in language_model.
    prefix = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in state else 'model.'
    w = {}
    def take(name, dtype=torch.float16):
        from .device import selected_accelerator
        if state[prefix + name].device.type != 'meta' and selected_accelerator() == 'rocm':
            from .rocm_policy import require_host_headroom
            require_host_headroom()
        return state.pop(prefix + name).to(dtype)
    w['embed'] = take('embed_tokens.weight')
    w['norm'] = take('norm.weight', torch.float32) + 1
    w['head'] = w['embed'] if config['tie_word_embeddings'] else state.pop('lm_head.weight').half()
    for i, kind in enumerate(config['layer_types']):
        layer = f'layers.{i}.'
        w[f'{i}.norm1'] = take(layer+'input_layernorm.weight', torch.float32) + 1
        w[f'{i}.norm2'] = take(layer+'post_attention_layernorm.weight', torch.float32) + 1
        w[f'{i}.gate_up'] = torch.cat([take(layer+f'mlp.{name}_proj.weight') for name in ('gate', 'up')])
        w[f'{i}.down'] = take(layer+'mlp.down_proj.weight')
        if kind == 'full_attention':
            w[f'{i}.qkv'] = torch.cat([take(layer+f'self_attn.{name}_proj.weight') for name in ('q', 'k', 'v')])
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
        return ((accelerator == 'rocm' or admitted_a30_q4) and self.fused is not None
                and self.chunk_scan is None)

    @property
    def requires_full_prefill(self):
        return not self.supports_chunked_prefill

    def __init__(self, model, use_graph=True, max_batch=4, kernels=None):
        if not 1 <= max_batch <= 16:
            raise ValueError('batch capacity must be between 1 and 16')
        from .layer_stream import enabled as offload_enabled, validate as validate_offload
        self.offload = None
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
        selected = kernels or os.environ.get('DLLM_KERNELS', 'auto')
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
        metal_scan = os.environ.get('DLLM_HYBRID_METAL_SCAN','auto')
        if metal_scan not in ('auto','0','1'):
            raise ValueError('DLLM_HYBRID_METAL_SCAN must be auto, 0 or 1')
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
            raise ValueError('hybrid packed weights require native GPU or qualified Apple Metal kernels')
        self.chunk_scan = None
        prefill_backend = os.environ.get('DLLM_HYBRID_PREFILL', 'scan')
        if prefill_backend not in ('scan', 'fla'):
            raise ValueError('DLLM_HYBRID_PREFILL must be scan or fla')
        if prefill_backend == 'fla':
            if not self.fused:
                raise ValueError('FLA prefill requires the CUDA Triton backend')
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            self.chunk_scan = chunk_gated_delta_rule
            self.kernel_backend += '+fla-prefill'
        if self.fused:
            from . import kernels as dense_kernels
            self.dense_fused = dense_kernels
        from .capacity import hybrid_state_bytes
        state_bytes = hybrid_state_bytes(c, self.context, max_batch, cache_element_bytes)
        empty_cache()
        free = available_memory()
        # Intel eager projections store FP32 matrices; host byte counts are
        # insufficient because the source mapping retains FP16 matrices.
        device_weight_bytes = (sum(t.numel()*4 for t in
            {id(t): t for t in model.weights.values()}.values())
            if self.device == 'xpu' else model.bytes)
        layout = None
        if streaming:
            from .layer_stream import plan
            layout = plan(model.weights, c['num_hidden_layers'])
            device_weight_bytes = layout.peak_bytes
        # Eager output logits explicitly promote the full output head to FP32.
        # Streaming must budget that temporary in addition to resident weights.
        self.output_head_workspace_bytes = (model.weights['head'].numel()*4
            if streaming and model.weights['head'].dtype != torch.float32 else 0)
        self.required_memory_bytes = (device_weight_bytes + state_bytes + 512*1024**2
                                      + self.output_head_workspace_bytes)
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
                        dtype=torch.float32 if self.device == 'xpu' else value.dtype)
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
        for i, kind in enumerate(c['layer_types']):
            if kind == 'full_attention':
                self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=self.compute_dtype, device=self.device)
            else:
                self.conv[i] = torch.zeros((max_batch, self.conv_width, c['linear_conv_kernel_dim']), dtype=self.compute_dtype, device=self.device)
                self.state[i] = torch.zeros((max_batch, self.lv, self.dk, self.dv), dtype=torch.float32, device=self.device)
        dim = c['rotary_dim']
        inv = c['rope_theta'] ** (-torch.arange(0, dim, 2, device=self.device, dtype=torch.float32)/dim)
        angles = torch.outer(torch.arange(self.context, device=self.device), inv)
        angles = torch.cat((angles, angles), -1)
        self.cos, self.sin = angles.cos(), angles.sin()
        self.arange = torch.arange(self.context, device=self.device)
        if self.fused:
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
                    print(f'dllm: hybrid kernel toolchain unavailable ({type(error).__name__}); using torch kernels', file=sys.stderr)
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
        for tensor in (*self.conv.values(), *self.state.values(), *self.kv.values()):
            tensor.zero_()

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
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps) * weight).to(dtype)

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
            q, k = [a.float()*torch.rsqrt(a.float().square().sum(-1, keepdim=True)+1.e-6) for a in (q, k)]
            q, k = [a.repeat_interleave(self.lv//self.lk, dim=2) for a in (q, k)]
            q = q * (self.dk**-.5)
            if self.metal_scan is not None:
                out=self.metal_scan(q,k,v,decay,beta,self.state[i],slots,prefill)
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
        ck, cv = self.kv[i].unbind(0)
        ck[slots[:, None], :, positions, :] = k
        cv[slots[:, None], :, positions, :] = v
        from .grouped_attention import use_grouped_prefill
        from .device import selected_accelerator
        end = prefix_length + length
        if use_grouped_prefill(selected_accelerator(), self.hybrid_precision, prefill, length, end):
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
        elif self.fused:
            out = self.dense_fused.attention(q.transpose(1, 2), ck, cv, positions[:, 0].contiguous(), slots)
        else:
            mask = self.arange.view(1, 1, 1, -1) <= positions.view(n, 1, 1, 1)
            out = scaled_attention(q.transpose(1, 2), ck.index_select(0, slots), cv.index_select(0, slots),
                                                 attn_mask=mask, enable_gqa=True)
        out = out.transpose(1, 2).float()*gate.float().sigmoid()
        return self.linear(out.reshape(n, length, -1).to(self.compute_dtype), w[f'{i}.o'])

    @torch.inference_mode()
    def forward(self, ids, positions, slots, prefill, prefix_length=0, return_logits=True):
        return self.forward_embeddings(self.embedding(ids),positions,slots,prefill,
                                       prefix_length, return_logits)

    def embedding(self, ids):
        return F.embedding(ids, self.w['embed']).float()

    def forward_embeddings(self, x, positions, slots, prefill, prefix_length=0, return_logits=True):
        """Shared text backbone after token or image/text embedding assembly."""
        x = self.forward_layers(x, positions, slots, prefill, prefix_length)
        return self.output_logits(x[:, -1]) if return_logits else None

    def forward_layers(self, x, positions, slots, prefill, prefix_length=0,
                       layer_start=0, layer_end=None):
        """Execute a contiguous range using global layer indices and positions.

        Returns every hidden position. The caller owns embedding, final-head
        selection and request cursor commit across all participating ranges.
        """
        if layer_end is None:
            layer_end = self.c['num_hidden_layers']
        if (type(layer_start) is not int or type(layer_end) is not int
                or not 0 <= layer_start < layer_end <= self.c['num_hidden_layers']):
            raise ValueError('invalid hybrid layer range')
        for i in range(layer_start, layer_end):
            kind = self.c['layer_types'][i]
            if self.offload is not None:
                self.offload.activate(i)
            from .device import selected_accelerator
            if selected_accelerator() == 'rocm':
                from .rocm_policy import require_host_headroom
                require_host_headroom()
            h = self.norm(x, self.w[f'{i}.norm1'])
            x = x + (self.linear_attention(h, i, slots, prefill, prefix_length) if kind == 'linear_attention'
                     else self.attention(h, i, positions, slots, prefill, prefix_length))
            gate_up = self.linear(self.norm(x, self.w[f'{i}.norm2']), self.w[f'{i}.gate_up'])
            if self.fused:
                activated = self.dense_fused.swiglu(gate_up.reshape(-1, gate_up.shape[-1]), True)
                activated = activated.view(*gate_up.shape[:-1], -1)
            else:
                gate, up = gate_up.chunk(2, -1)
                activated = (F.silu(gate.float())*up.float()).to(self.compute_dtype)
            x = x + self.linear(activated, self.w[f'{i}.down'])
        return x

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
        return result

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
        return logits

    def decode(self, token):
        result = self.decode_batch([token], [0])
        self.length = self.lengths[0]
        return result
