"""Isolated OLMoE packed short-prefill experiment; attention remains slot-local."""
import hashlib
from itertools import accumulate
from pathlib import Path
import torch
from torch.nn import functional as F


def install_validation_dispatch():
    from damendaemon.moe import OlmoeExecutor
    original = OlmoeExecutor.prefill_batch
    report = dict(kind='experimental packed OLMoE short prefill', selected_calls=0,
                  fallback_calls=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        if (not prompts or len(prompts) != len(slots) or len(set(slots)) != len(slots)
                or any(slot < 0 or slot >= self.max_batch for slot in slots)
                or any(not ids or len(ids) > self.context for ids in prompts)):
            raise ValueError('invalid prefill batch')
        lengths = list(map(len, prompts))
        if not self.grouped_prefill or len(prompts) == 1 or sum(lengths) > 128:
            report['fallback_calls'] += 1
            return original(self, prompts, slots)
        report['selected_calls'] += 1
        ids = torch.tensor([token for prompt in prompts for token in prompt], device=self.device)
        positions = torch.tensor([position for length in lengths for position in range(length)], device=self.device)
        slot_ids = torch.tensor([slot for slot, length in zip(slots, lengths) for _ in range(length)], device=self.device)
        w = self.w
        x = F.embedding(ids, w['embed']).float()
        for layer in range(self.c['num_hidden_layers']):
            packed = self._linear(self.norm(x, w[f'{layer}.norm1']), w[f'{layer}.qkv.weight'])
            q, k, v = packed.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
            q = self.norm(q.contiguous(), w[f'{layer}.q_norm'])
            k = self.norm(k.contiguous(), w[f'{layer}.k_norm'])
            ck, cv = self.kv[layer].unbind(0)
            query = self.kernels.rope_batch(torch.cat((q, k, v), dim=-1), self.cos, self.sin,
                                            positions, slot_ids, ck, cv, self.heads, precise=True)
            attended, offset = [], 0
            for slot, length in zip(slots, lengths):
                q = query[offset:offset+length, :, 0, :].transpose(0, 1).unsqueeze(0)
                value = self.prefill_attention(q, ck[slot:slot+1, :, :length], cv[slot:slot+1, :, :length])
                attended.append(value.transpose(1, 2).reshape(length, -1))
                offset += length
            x = x+self._linear(torch.cat(attended), w[f'{layer}.out'])
            x = x+self.experts(self.norm(x, w[f'{layer}.norm2']), layer, True)
        ends = torch.tensor([end-1 for end in accumulate(lengths)], device=self.device)
        logits = self._linear(self.norm(x.index_select(0, ends), w['norm']), w['head'])
        for slot, length in zip(slots, lengths):
            self.lengths[slot] = length
        return logits

    OlmoeExecutor.prefill_batch = prefill_batch
    return report
