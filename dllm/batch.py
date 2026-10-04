"""Slot-isolated batched decode with one CUDA graph per admitted batch size."""
import ctypes
from itertools import accumulate

import torch
from torch.nn import functional as F

from .device import attention as scaled_attention

from .qwen2 import Executor


class BatchExecutor(Executor):
    def __init__(self, model, use_graph=True, max_batch=4, kernels=None):
        if not 1 <= max_batch <= 16:
            raise ValueError("batch capacity must be between 1 and 16")
        super().__init__(model, use_graph=False, kernels=kernels, max_batch=max_batch)
        self.lengths = [0] * max_batch
        self.batch_graphs = {}
        if use_graph and self.device == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for size in range(1, max_batch + 1):
                metadata = torch.zeros((3, size), dtype=torch.long, device=self.device)
                ids, positions, slots = metadata.unbind(0)
                slots.copy_(torch.arange(size, device=self.device))
                staging = torch.empty((3, size), dtype=torch.long, pin_memory=True)
                staging_values = (ctypes.c_int64 * (3 * size)).from_address(staging.data_ptr())
                copied = torch.cuda.Event()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.inference_mode():
                    for _ in range(3):
                        self.forward_batch(ids, positions, slots)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    logits = self.forward_batch(ids, positions, slots)
                    next_ids = logits.argmax(dim=-1)
                self.batch_graphs[size] = (metadata, staging, staging_values, copied, graph, logits, next_ids)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        batch = ids.shape[0]
        w = self.w
        x = F.embedding(ids, w["embed"]).to(self.residual_dtype)
        if not self.kernels:
            cos = self.cos[positions].view(batch, 1, 1, self.dim)
            sin = self.sin[positions].view(batch, 1, 1, self.dim)
            mask = self.arange.view(1, 1, 1, -1) <= positions.view(batch, 1, 1, 1)
        for i in range(self.c["num_hidden_layers"]):
            qkv = self.linear(self.norm(x, w[f"{i}.norm1"]), w[f"{i}.qkv.weight"], w.get(f"{i}.qkv.bias"))
            cache_k, cache_v = self.kv[i].unbind(0)
            if self.kernels:
                q = self.kernels.rope_batch(qkv, self.cos, self.sin, positions, slots,
                                             cache_k, cache_v, self.heads,
                    w.get(f"{i}.q_norm"), w.get(f"{i}.k_norm"), self.eps)
                attended = self.kernels.attention(q, cache_k, cache_v, positions, slots)
            else:
                q, k, v = qkv.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
                q, k = self.normalize_qk(q, k, i)
                q = self.rotate(q.view(batch, self.heads, 1, self.dim), cos, sin)
                k = self.rotate(k.view(batch, self.kv_heads, 1, self.dim), cos, sin)
                v = v.view(batch, self.kv_heads, self.dim)
                cache_k[slots, :, positions, :] = k.squeeze(2)
                cache_v[slots, :, positions, :] = v
                attended = scaled_attention(q, cache_k.index_select(0, slots),
                    cache_v.index_select(0, slots), attn_mask=mask, enable_gqa=True)
            x = x + self.linear(attended.reshape(batch, -1), w[f"{i}.out"])
            gate_up = self.linear(self.norm(x, w[f"{i}.norm2"]), w[f"{i}.gate_up"])
            if self.kernels:
                activated = self.kernels.swiglu(gate_up, self.precise)
            else:
                gate, up = gate_up.chunk(2, -1)
                activated = self.activate(gate, up)
            x = x + self.linear(activated, w[f"{i}.down"])
        return self.linear(self.norm(x, w["norm"]), w["head"])

    def prefill(self, tokens, slot=0):
        result = super().prefill(tokens, slot)
        self.lengths[slot] = len(tokens)
        return result

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        """Pack different prompt lengths into shared linear layers; isolate attention."""
        if (not prompts or len(prompts) != len(slots) or len(set(slots)) != len(slots)
                or any(slot < 0 or slot >= self.max_batch for slot in slots)
                or any(not ids or len(ids) > self.context for ids in prompts)):
            raise ValueError("invalid prefill batch")
        if len(prompts) == 1:
            return self.prefill(prompts[0], slots[0])
        lengths = [len(ids) for ids in prompts]
        ids = torch.tensor([token for prompt in prompts for token in prompt], device=self.device)
        positions = torch.tensor([i for length in lengths for i in range(length)], device=self.device)
        slot_ids = torch.tensor([slot for slot, length in zip(slots, lengths) for _ in range(length)], device=self.device)
        w = self.w
        x = F.embedding(ids, w["embed"]).to(self.residual_dtype)
        if not self.kernels:
            cos = self.cos[positions].view(-1, 1, 1, self.dim)
            sin = self.sin[positions].view(-1, 1, 1, self.dim)
        for i in range(self.c["num_hidden_layers"]):
            qkv = self.linear(self.norm(x, w[f"{i}.norm1"]), w[f"{i}.qkv.weight"], w.get(f"{i}.qkv.bias"))
            cache_k, cache_v = self.kv[i].unbind(0)
            if self.kernels:
                query = self.kernels.rope_batch(qkv, self.cos, self.sin, positions, slot_ids,
                                                 cache_k, cache_v, self.heads,
                    w.get(f"{i}.q_norm"), w.get(f"{i}.k_norm"), self.eps)
            else:
                q, k, v = qkv.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
                q, k = self.normalize_qk(q, k, i)
                query = self.rotate(q.view(-1, self.heads, 1, self.dim), cos, sin)
                k = self.rotate(k.view(-1, self.kv_heads, 1, self.dim), cos, sin)
                cache_k[slot_ids, :, positions, :] = k.squeeze(2)
                cache_v[slot_ids, :, positions, :] = v.view(-1, self.kv_heads, self.dim)
            attended, offset = [], 0
            for slot, length in zip(slots, lengths):
                q = query[offset:offset+length, :, 0, :].transpose(0, 1).unsqueeze(0)
                attention = scaled_attention(q, cache_k[slot:slot+1, :, :length],
                    cache_v[slot:slot+1, :, :length], is_causal=True, enable_gqa=True)
                attended.append(attention.transpose(1, 2).reshape(length, -1))
                offset += length
            x = x + self.linear(torch.cat(attended), w[f"{i}.out"])
            gate_up = self.linear(self.norm(x, w[f"{i}.norm2"]), w[f"{i}.gate_up"])
            if self.kernels:
                activated = self.kernels.swiglu(gate_up, self.precise)
            else:
                gate, up = gate_up.chunk(2, -1)
                activated = self.activate(gate, up)
            x = x + self.linear(activated, w[f"{i}.down"])
        ends = torch.tensor([end - 1 for end in accumulate(lengths)], device=self.device)
        logits = self.linear(self.norm(x.index_select(0, ends), w["norm"]), w["head"])
        for slot, length in zip(slots, lengths):
            self.lengths[slot] = length
        return logits

    @torch.inference_mode()
    def decode_batch(self, tokens, slots):
        if (not tokens or len(tokens) != len(slots) or len(set(slots)) != len(slots)
                or any(slot < 0 or slot >= self.max_batch for slot in slots)):
            raise ValueError("invalid batch slots")
        positions = [self.lengths[slot] for slot in slots]
        if any(position <= 0 or position >= self.context for position in positions):
            raise ValueError("batch cache is empty or exhausted")
        if len(tokens) in self.batch_graphs:
            metadata, staging, staging_values, copied, graph, logits, next_ids = self.batch_graphs[len(tokens)]
            # Do not overwrite pinned memory until the previous DMA has read it.
            copied.synchronize()
            staging_values[:] = list(tokens) + positions + list(slots)
            metadata.copy_(staging, non_blocking=True)
            copied.record()
            graph.replay()
        else:
            # Transfer one metadata buffer instead of synchronizing three small
            # host-to-device tensor constructions on every eager decode step.
            metadata = torch.tensor([tokens, positions, slots], device=self.device)
            logits = self.forward_batch(*metadata.unbind(0))
            next_ids = logits.argmax(-1)
        for slot in slots:
            self.lengths[slot] += 1
        return logits, next_ids
