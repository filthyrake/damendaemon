"""Isolated offset-prefill experiment for qualified A2 Q8 OLMoE."""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F


def install_validation_dispatch():
    from dllm.moe import OlmoeExecutor
    report = dict(kind='experimental offset-aware OLMoE prefill', selected_calls=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @torch.inference_mode()
    def prefill_chunk(self, tokens, slot, start, final=True):
        if not self.grouped_prefill:
            raise ValueError('chunked OLMoE experiment requires qualified A2 Q8 geometry')
        if (type(start) is not int or start < 0 or not 0 <= slot < self.max_batch
                or (start and self.lengths[slot] != start)):
            raise ValueError('invalid partial prompt cache state')
        if not tokens or start+len(tokens) > self.context:
            raise ValueError('prompt is empty or exceeds context')
        count, end = len(tokens), start+len(tokens)
        ids = torch.tensor(tokens, device=self.device)
        positions = torch.arange(start, end, device=self.device)
        slots = torch.full_like(positions, slot)
        w = self.w
        x = F.embedding(ids, w['embed']).float()
        for layer in range(self.c['num_hidden_layers']):
            packed = self._linear(self.norm(x, w[f'{layer}.norm1']), w[f'{layer}.qkv.weight'])
            q, k, v = packed.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
            q = self.norm(q.contiguous(), w[f'{layer}.q_norm'])
            k = self.norm(k.contiguous(), w[f'{layer}.k_norm'])
            ck, cv = self.kv[layer].unbind(0)
            query = self.kernels.rope_batch(torch.cat((q,k,v), dim=-1), self.cos, self.sin,
                                            positions, slots, ck, cv, self.heads, precise=True)
            attended = self.prefill_attention(query[:,:,0,:].transpose(0,1).unsqueeze(0),
                                              ck[slot:slot+1,:,:end], cv[slot:slot+1,:,:end],
                                              query_start=start).transpose(1,2).reshape(count,-1)
            x = x+self._linear(attended, w[f'{layer}.out'])
            x = x+self.experts(self.norm(x, w[f'{layer}.norm2']), layer, True)
        logits = self._linear(self.norm(x[-1:], w['norm']), w['head']) if final else None
        self.lengths[slot] = end
        self.length = end
        report['selected_calls'] += 1
        return logits

    OlmoeExecutor.supports_chunked_prefill = property(lambda self: self.grouped_prefill)
    OlmoeExecutor.prefill_chunk = prefill_chunk
    return report
