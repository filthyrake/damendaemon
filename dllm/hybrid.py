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

from .device import selected_device, empty_cache, available_memory, attention as scaled_attention

from .batch import BatchExecutor


def load_weights(state, config):
    # Multimodal checkpoints wrap the text backbone in language_model.
    prefix = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in state else 'model.'
    w = {}
    def take(name, dtype=torch.float16):
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

    def __init__(self, model, use_graph=True, max_batch=4, kernels=None):
        if not 1 <= max_batch <= 16:
            raise ValueError('batch capacity must be between 1 and 16')
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
        full = c['layer_types'].count('full_attention')
        linear = c['num_hidden_layers'] - full
        state_bytes = max_batch * (full * 2*self.kh*self.context*self.dim*2
            + linear * (self.lv*self.dk*self.dv*4 + self.conv_width*c['linear_conv_kernel_dim']*2))
        empty_cache()
        free = available_memory()
        if model.bytes + state_bytes + 512*1024**2 > free:
            raise MemoryError('insufficient GPU memory for hybrid weights, state, and workspace')
        copies = {}
        self.w = {}
        for key, value in model.weights.items():
            if id(value) not in copies:
                copies[id(value)] = value.to(self.device)
            self.w[key] = copies[id(value)]
        self.kv, self.conv, self.state = {}, {}, {}
        for i, kind in enumerate(c['layer_types']):
            if kind == 'full_attention':
                self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=torch.float16, device=self.device)
            else:
                self.conv[i] = torch.zeros((max_batch, self.conv_width, c['linear_conv_kernel_dim']), dtype=torch.float16, device=self.device)
                self.state[i] = torch.zeros((max_batch, self.lv, self.dk, self.dv), dtype=torch.float32, device=self.device)
        dim = c['rotary_dim']
        inv = c['rope_theta'] ** (-torch.arange(0, dim, 2, device=self.device, dtype=torch.float32)/dim)
        angles = torch.outer(torch.arange(self.context, device=self.device), inv)
        angles = torch.cat((angles, angles), -1)
        self.cos, self.sin = angles.cos(), angles.sin()
        self.arange = torch.arange(self.context, device=self.device)
        if self.fused:
            probe = torch.zeros(1, dtype=torch.long, device=self.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                try:
                    self.forward_batch(probe, probe, probe)
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError) as error:
                    if selected != 'auto':
                        raise
                    if self.chunk_scan is not None:
                        raise
                    self.fused = None
                    self.kernel_backend = 'torch-hybrid'
                    print(f'dllm: hybrid kernel toolchain unavailable ({type(error).__name__}); using torch kernels', file=sys.stderr)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        self.batch_graphs = {}
        if use_graph:
            for size in range(1, max_batch+1):
                metadata = torch.zeros((3, size), dtype=torch.long, device=self.device)
                ids, positions, slots = metadata.unbind(0)
                slots.copy_(torch.arange(size, device=self.device))
                staging = torch.empty((3, size), dtype=torch.long, pin_memory=True)
                values = (ctypes.c_int64 * (3*size)).from_address(staging.data_ptr())
                copied = torch.cuda.Event()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.inference_mode():
                    for _ in range(3):
                        self.forward_batch(ids, positions, slots)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    logits = self.forward_batch(ids, positions, slots)
                    next_ids = logits.argmax(-1)
                self.batch_graphs[size] = (metadata, staging, values, copied, graph, logits, next_ids)
        # Capturing recurrent updates changes state. No captured warmup is user history.
        for tensor in (*self.conv.values(), *self.state.values(), *self.kv.values()):
            tensor.zero_()

    def norm(self, x, weight, dtype=torch.float16):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps) * weight).to(dtype)

    def rotate(self, x, positions):
        dim = self.c['rotary_dim']
        rotated, rest = x[..., :dim], x[..., dim:]
        left, right = rotated.chunk(2, -1)
        cos, sin = self.cos[positions].unsqueeze(-2), self.sin[positions].unsqueeze(-2)
        rotated = rotated*cos + torch.cat((-right, left), -1)*sin
        return torch.cat((rotated, rest), -1).half()

    def linear_attention(self, x, i, slots, prefill):
        w = self.w
        n, length, _ = x.shape
        qkv = F.linear(x, w[f'{i}.qkv'])
        z = F.linear(x, w[f'{i}.z']).view(n, length, self.lv, self.dv)
        beta = F.linear(x, w[f'{i}.b']).float().sigmoid()
        decay = -w[f'{i}.A'].exp()*F.softplus(F.linear(x, w[f'{i}.a']).float()+w[f'{i}.dt'])
        if prefill:
            history = qkv.transpose(1, 2).contiguous()
            width = self.c['linear_conv_kernel_dim']
            self.conv[i][slots] = F.pad(history, (max(0, width-length), 0))[..., -width:]
            convolved = F.conv1d(history.float(), w[f'{i}.conv'].float(),
                padding=width-1, groups=self.conv_width)[..., :length]
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
            out = self.fused.scan(q, k, v, decay, beta, self.state[i], slots, reset=prefill)
        else:
            q, k = [a.float()*torch.rsqrt(a.float().square().sum(-1, keepdim=True)+1.e-6) for a in (q, k)]
            q, k = [a.repeat_interleave(self.lv//self.lk, dim=2) for a in (q, k)]
            q = q * (self.dk**-.5)
            state = torch.zeros_like(self.state[i].index_select(0, slots)) if prefill else self.state[i].index_select(0, slots)
            outputs = []
            for t in range(length):
                state = state * decay[:, t].exp()[..., None, None]
                delta = (v[:, t].float()-(state*k[:, t, :, :, None]).sum(-2))*beta[:, t, :, None]
                state = state + k[:, t, :, :, None]*delta.unsqueeze(-2)
                outputs.append((state*q[:, t, :, :, None]).sum(-2))
            self.state[i][slots] = state
            out = torch.stack(outputs, 1)
        out = self.norm(out, w[f'{i}.gated_norm'], torch.float32) * F.silu(z.float())
        return F.linear(out.reshape(n, length, -1).half(), w[f'{i}.out'])

    def attention(self, x, i, positions, slots, prefill):
        n, length, _ = x.shape
        w = self.w
        q, k, v = F.linear(x, w[f'{i}.qkv']).split((2*self.heads*self.dim, self.kh*self.dim, self.kh*self.dim), -1)
        q, gate = q.view(n, length, self.heads, 2*self.dim).chunk(2, -1)
        k, v = [a.view(n, length, self.kh, self.dim) for a in (k, v)]
        q = self.rotate(self.norm(q, w[f'{i}.q_norm'], torch.float32), positions)
        k = self.rotate(self.norm(k, w[f'{i}.k_norm'], torch.float32), positions)
        ck, cv = self.kv[i].unbind(0)
        ck[slots[:, None], :, positions, :] = k
        cv[slots[:, None], :, positions, :] = v
        if prefill:
            out = scaled_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                 is_causal=True, enable_gqa=True)
        elif self.fused:
            out = self.dense_fused.attention(q.transpose(1, 2), ck, cv, positions[:, 0].contiguous(), slots)
        else:
            mask = self.arange.view(1, 1, 1, -1) <= positions.view(n, 1, 1, 1)
            out = scaled_attention(q.transpose(1, 2), ck.index_select(0, slots), cv.index_select(0, slots),
                                                 attn_mask=mask, enable_gqa=True)
        out = out.transpose(1, 2).float()*gate.float().sigmoid()
        return F.linear(out.reshape(n, length, -1).half(), w[f'{i}.o'])

    @torch.inference_mode()
    def forward(self, ids, positions, slots, prefill):
        x = F.embedding(ids, self.w['embed']).float()
        for i, kind in enumerate(self.c['layer_types']):
            h = self.norm(x, self.w[f'{i}.norm1'])
            x = x + (self.linear_attention(h, i, slots, prefill) if kind == 'linear_attention'
                     else self.attention(h, i, positions, slots, prefill))
            gate, up = F.linear(self.norm(x, self.w[f'{i}.norm2']), self.w[f'{i}.gate_up']).chunk(2, -1)
            x = x + F.linear((F.silu(gate.float())*up.float()).half(), self.w[f'{i}.down'])
        normalized = self.norm(x[:, -1], self.w['norm'])
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

    def decode(self, token):
        result = self.decode_batch([token], [0])
        self.length = self.lengths[0]
        return result
