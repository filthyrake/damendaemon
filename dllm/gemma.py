"""Independent Gemma 3 text decoder with local/global attention.

Shares residency, cache slots and CUDA graph replay with the dense executor.
Prefill uses SDPA; CUDA decode uses live-prefix/window attention. Matrix
operations preserve FP32 arithmetic while optionally reading FP16 weights.
"""
import math
import torch
from torch.nn import functional as F

from .batch import BatchExecutor
from .device import attention
from .rotary import inverse_frequencies


def load_weights(state, config):
    def take(name):
        return state.pop(name).to(torch.bfloat16)
    def norm(name):
        return state.pop(name).float() + 1
    weights = {'embed': take('model.embed_tokens.weight'), 'norm': norm('model.norm.weight')}
    weights['head'] = weights['embed'] if config['tie_word_embeddings'] else take('lm_head.weight')
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        weights[f'{i}.qkv.weight'] = torch.cat([take(prefix + f'self_attn.{p}_proj.weight') for p in ('q','k','v')])
        weights[f'{i}.out'] = take(prefix + 'self_attn.o_proj.weight')
        weights[f'{i}.gate_up'] = torch.cat([take(prefix + f'mlp.{p}_proj.weight') for p in ('gate','up')])
        weights[f'{i}.down'] = take(prefix + 'mlp.down_proj.weight')
        for short, name in (('norm1','input_layernorm'), ('norm2','post_attention_layernorm'),
                            ('norm3','pre_feedforward_layernorm'), ('norm4','post_feedforward_layernorm'),
                            ('q_norm','self_attn.q_norm'), ('k_norm','self_attn.k_norm')):
            weights[f'{i}.{short}'] = norm(prefix + name + '.weight')
    return weights


class GemmaExecutor(BatchExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_backend = (("metal" if self.device == "mps" else "triton") if self.kernels else "torch") + "-gemma"

    def _tables(self):
        # Base construction warms virtual forward before batch graph capture.
        if hasattr(self, 'local_cos'):
            return
        if self.device == 'cuda':
            self.linear = self._linear
        inv = inverse_frequencies({**self.c, 'rope_theta': self.c['rope_local_base_freq']}, self.device)
        angles = torch.outer(self.arange.float(), inv)
        angles = torch.cat((angles, angles), -1)
        self.local_cos, self.local_sin = angles.cos().to(self.residual_dtype), angles.sin().to(self.residual_dtype)
        self.embed_scale = torch.tensor(math.sqrt(self.c['hidden_size']), device=self.device, dtype=self.compute_dtype)

    def _linear(self, x, weight):
        if self.kernels:
            return self.kernels.linear_fp32(x, weight)
        return F.linear(x.float(), weight.float())

    def norm(self, x, weight):
        if self.kernels:
            return self.kernels.rms(x.contiguous(), weight, self.eps, True)
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * weight).to(self.compute_dtype)

    @torch.inference_mode()
    def forward(self, ids, positions, prefill, slot=0):
        slots = torch.full_like(positions, slot)
        return self._forward(ids, positions, slots, prefill)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        return self._forward(ids, positions, slots, False)

    def _forward(self, ids, positions, slots, prefill):
        self._tables()
        n, w = ids.numel(), self.w
        x = F.embedding(ids, w['embed']).to(self.compute_dtype) * self.embed_scale
        x = x.to(self.residual_dtype)
        keys = positions if prefill else self.arange
        mask = keys.view(1, -1) <= positions.view(-1, 1)
        local_mask = mask & (keys.view(1, -1) > positions.view(-1, 1) - self.c['sliding_window'])
        for i, layer in enumerate(self.c['layer_types']):
            local = layer == 'sliding_attention'
            cos, sin = (self.local_cos, self.local_sin) if local else (self.cos, self.sin)
            cos, sin = cos[positions, None, :], sin[positions, None, :]
            packed = self.linear(self.norm(x, w[f'{i}.norm1']), w[f'{i}.qkv.weight'])
            q, k, v = packed.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
            q = self.rotate(self.norm(q.view(n, self.heads, self.dim), w[f'{i}.q_norm']), cos, sin)
            k = self.rotate(self.norm(k.view(n, self.kv_heads, self.dim), w[f'{i}.k_norm']), cos, sin)
            v = v.view(n, self.kv_heads, self.dim)
            cache_k, cache_v = self.kv[i].unbind(0)
            cache_k[slots, :, positions, :] = k
            cache_v[slots, :, positions, :] = v
            selected_mask = local_mask if local else mask
            if prefill:
                attended = attention(q.transpose(0,1).unsqueeze(0), k.transpose(0,1).unsqueeze(0),
                    v.transpose(0,1).unsqueeze(0), attn_mask=selected_mask, enable_gqa=True,
                    scale=self.c['query_pre_attn_scalar'] ** -.5)
                attended = attended.squeeze(0).transpose(0,1).reshape(n, -1)
            elif self.device == 'cuda' and self.kernels:
                attended = self.kernels.attention(q.unsqueeze(2).contiguous(), cache_k, cache_v,
                    positions, slots, window=self.c['sliding_window'] if local else 0,
                    scale=self.c['query_pre_attn_scalar'] ** -.5).reshape(n, -1)
            else:
                attended = attention(q.unsqueeze(2), cache_k.index_select(0, slots), cache_v.index_select(0, slots),
                    attn_mask=selected_mask[:,None,None,:], enable_gqa=True,
                    scale=self.c['query_pre_attn_scalar'] ** -.5).reshape(n, -1)
            x = x + self.norm(self.linear(attended, w[f'{i}.out']), w[f'{i}.norm2'])
            gate, up = self.linear(self.norm(x, w[f'{i}.norm3']), w[f'{i}.gate_up']).chunk(2, -1)
            hidden = F.gelu(gate.float(), approximate='tanh') * up.float()
            x = x + self.norm(self.linear(hidden, w[f'{i}.down']), w[f'{i}.norm4'])
        if prefill:
            x = x[-1:]
        return self.linear(self.norm(x, w['norm']), w['head'])

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        if (not prompts or len(prompts) != len(slots) or len(set(slots)) != len(slots)
                or any(slot < 0 or slot >= self.max_batch for slot in slots)
                or any(not ids or len(ids) > self.context for ids in prompts)):
            raise ValueError('invalid prefill batch')
        return torch.cat([self.prefill(prompt, slot) for prompt, slot in zip(prompts, slots)])

    def decode(self, token):
        return self.decode_batch([token], [0])
