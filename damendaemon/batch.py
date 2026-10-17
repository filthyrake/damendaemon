"""Slot-isolated batched decode with one CUDA graph per admitted batch size."""
import ctypes
from itertools import accumulate

import torch
from torch.nn import functional as F

from .qwen2 import Executor
from .prefill import groups


class BatchExecutor(Executor):
    def __init__(self, model, use_graph=True, max_batch=4, kernels=None):
        if not 1 <= max_batch <= 16:
            raise ValueError("batch capacity must be between 1 and 16")
        super().__init__(model, use_graph=False, kernels=kernels, max_batch=max_batch)
        self.lengths = [0] * max_batch
        self.batch_graphs = {}
        self.configure_prefill_graphs(use_graph)
        if use_graph and self.device == "cuda":
            self.capture_graphs()
        from .speculative import create
        self.speculative_verifier=create(self,use_graph)

    def capture_graphs(self):
        """Capture decode against executor-owned stable metadata and cache buffers."""
        if self.device == 'cuda':
            from .cuda_warmup import warmup_stream
            for size in range(1, self.max_batch + 1):
                metadata = torch.zeros((3, size), dtype=torch.long, device=self.device)
                ids, positions, slots = metadata.unbind(0)
                slots.copy_(torch.arange(size, device=self.device))
                staging = torch.empty((3, size), dtype=torch.long, pin_memory=True)
                staging_values = (ctypes.c_int64 * (3 * size)).from_address(staging.data_ptr())
                copied = torch.cuda.Event()
                with warmup_stream(), torch.inference_mode():
                    for _ in range(3):
                        self.forward_batch(ids, positions, slots)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                # A previous encoder graph can leave RNG bookkeeping as inference
                # tensors. Enter inference mode before capture_begin touches it.
                with torch.inference_mode(), torch.cuda.graph(graph):
                    logits = self.forward_batch(ids, positions, slots)
                    next_ids = logits.argmax(dim=-1)
                self.batch_graphs[size] = (metadata, staging, staging_values, copied, graph, logits, next_ids)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        hidden=self.forward_batch_hidden(self.embedding(ids),positions,slots)
        return self.output_logits(hidden)

    @torch.inference_mode()
    def forward_batch_hidden(self, x, positions, slots):
        batch = x.shape[0]
        w = self.w
        if not self.kernels:
            cos = self.cos[positions].view(batch, 1, 1, self.dim)
            sin = self.sin[positions].view(batch, 1, 1, self.dim)
        for i in range(self.c["num_hidden_layers"]):
            qkv = self.linear(self.norm(x, w[f"{i}.norm1"]), w[f"{i}.qkv.weight"], w.get(f"{i}.qkv.bias"))
            cache_k, cache_v = self.kv[i].unbind(0)
            if self.kernels:
                q = self.kernels.rope_batch(qkv, self.cos, self.sin, positions, slots,
                                             cache_k, cache_v, self.heads,
                    w.get(f"{i}.q_norm"), w.get(f"{i}.k_norm"), self.eps, self.rotary_precise)
                attended = self.decode_attention(q, cache_k, cache_v, positions, slots)
            else:
                q, k, v = qkv.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
                q, k = self.normalize_qk(q, k, i)
                q = self.rotate(q.view(batch, self.heads, 1, self.dim), cos, sin)
                k = self.rotate(k.view(batch, self.kv_heads, 1, self.dim), cos, sin)
                v = v.view(batch, self.kv_heads, self.dim)
                cache_k[slots, :, positions, :] = k.squeeze(2)
                cache_v[slots, :, positions, :] = v
                attended = self.decode_attention(q, cache_k, cache_v, positions, slots)
            x = x + self.linear(attended.reshape(batch, -1), w[f"{i}.out"])
            gate_up = self.linear(self.norm(x, w[f"{i}.norm2"]), w[f"{i}.gate_up"])
            if self.kernels:
                activated = self.kernels.swiglu(gate_up, self.precise)
            else:
                gate, up = gate_up.chunk(2, -1)
                activated = self.activate(gate, up)
            x = x + self.linear(activated, w[f"{i}.down"])
        return x

    def prefill(self, tokens, slot=0):
        result = super().prefill(tokens, slot)
        self.lengths[slot] = len(tokens)
        return result

    def decode(self, token):
        # The base scalar graph is deliberately disabled in this constructor.
        # Reuse the captured one-row batch graph for direct/CLI callers too.
        if 1 not in self.batch_graphs:
            return super().decode(token)
        result = self.decode_batch([token], [0])
        self.length = self.lengths[0]
        return result

    @property
    def supports_chunked_prefill(self):
        return self.c['model_type'] in ('qwen2','qwen3','llama','mistral')

    @torch.inference_mode()
    def prefill_chunk(self,tokens,slot,start,final=True):
        if not self.supports_chunked_prefill:
            raise ValueError('chunked prefill is not supported for this architecture')
        if (type(start) is not int or start<0 or not 0<=slot<self.max_batch
                or (start and self.lengths[slot]!=start)):
            raise ValueError('invalid partial prompt cache state')
        return self._prefill_batch([tokens],[slot],[start],return_logits=final)

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        return self._prefill_batch(prompts,slots,[0]*len(prompts))

    def _prefill_batch(self,prompts,slots,starts,return_logits=True):
        """Pack different prompt lengths into shared linear layers; isolate attention."""
        if (not prompts or len(prompts) != len(slots) or len(set(slots)) != len(slots)
                or any(slot < 0 or slot >= self.max_batch for slot in slots)
                or any(not ids or start+len(ids) > self.context for ids,start in zip(prompts,starts))):
            raise ValueError("invalid prefill batch")
        token_budget = getattr(self, 'prefill_token_budget', self.context)
        if sum(map(len, prompts)) > token_budget:
            # Bound temporary activations by the prefill budget (one context
            # by default). Validate
            # the entire request set above before writing any group's cache.
            return torch.cat([self.prefill_batch(batch, assigned)
                              for batch, assigned in groups(prompts, slots, token_budget)])
        if len(prompts) == 1 and starts[0] == 0 and return_logits:
            return self.prefill(prompts[0], slots[0])
        lengths = [len(ids) for ids in prompts]
        ids = torch.tensor([token for prompt in prompts for token in prompt], device=self.device)
        positions = torch.tensor([i for length,start in zip(lengths,starts) for i in range(start,start+length)], device=self.device)
        slot_ids = torch.tensor([slot for slot, length in zip(slots, lengths) for _ in range(length)], device=self.device)
        x = self.forward_packed_hidden(self.embedding(ids),positions,slot_ids,lengths,slots,starts)
        logits=None
        if return_logits:
            ends = torch.tensor([end - 1 for end in accumulate(lengths)], device=self.device)
            logits = self.output_logits(x.index_select(0, ends))
        for slot, length, start in zip(slots, lengths, starts):
            self.lengths[slot] = start+length
        return logits

    @torch.inference_mode()
    def forward_packed_hidden(self,x,positions,slot_ids,lengths,slots,starts):
        """Shared packed linear work with attention isolated by prompt/slot."""
        w = self.w
        if not self.kernels:
            cos = self.cos[positions].view(-1, 1, 1, self.dim)
            sin = self.sin[positions].view(-1, 1, 1, self.dim)
        for i in range(self.c["num_hidden_layers"]):
            qkv = self.linear(self.norm(x, w[f"{i}.norm1"]), w[f"{i}.qkv.weight"], w.get(f"{i}.qkv.bias"))
            cache_k, cache_v = self.kv[i].unbind(0)
            if self.kernels:
                query = self.kernels.rope_batch(qkv, self.cos, self.sin, positions, slot_ids,
                                                 cache_k, cache_v, self.heads,
                    w.get(f"{i}.q_norm"), w.get(f"{i}.k_norm"), self.eps, self.rotary_precise)
            else:
                q, k, v = qkv.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
                q, k = self.normalize_qk(q, k, i)
                query = self.rotate(q.view(-1, self.heads, 1, self.dim), cos, sin)
                k = self.rotate(k.view(-1, self.kv_heads, 1, self.dim), cos, sin)
                cache_k[slot_ids, :, positions, :] = k.squeeze(2)
                cache_v[slot_ids, :, positions, :] = v.view(-1, self.kv_heads, self.dim)
            attended, offset = [], 0
            for slot, length, start in zip(slots, lengths, starts):
                q = query[offset:offset+length, :, 0, :].transpose(0, 1).unsqueeze(0)
                attention = self.prefill_attention(q, cache_k[slot:slot+1, :, :start+length],
                    cache_v[slot:slot+1, :, :start+length],query_start=start)
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
        return x

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
            logits = self.forward_batch(torch.tensor(tokens, device=self.device),
                torch.tensor(positions, device=self.device), torch.tensor(slots, device=self.device))
            next_ids = logits.argmax(-1)
        for slot in slots:
            self.lengths[slot] += 1
        return logits, next_ids
