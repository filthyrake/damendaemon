"""CUDA dense decoding over a shared, bounded pool of sixteen-token KV pages."""
from itertools import accumulate
import torch
from torch.nn import functional as F

from .batch import BatchExecutor
from .device import selected_device
from .pages import PagePool, PageCapacityError
from .prefill import groups
from .paged_kernels import copy_pages


class PagedExecutor(BatchExecutor):
    def __init__(self, model, use_graph=True, max_batch=4, kernels=None, cache_tokens=0):
        if selected_device() != 'cuda' or kernels == 'torch':
            raise ValueError('paged KV requires native CUDA kernels')
        self.pool = PagePool(cache_tokens, max_batch, model.max_context)
        self.kv_token_capacity = cache_tokens
        self.reserved_slots = set()
        self.page_table = torch.full((max_batch, (model.max_context+15)//16), -1,
                                     device='cuda', dtype=torch.int32)
        self.table_revision = -1
        # Scratch ownership makes graph warmup valid. It is released after capture.
        self.pool.resize_many([(slot,1) for slot in range(max_batch)])
        self._sync_pages()
        super().__init__(model, use_graph, max_batch, kernels='triton')
        for slot in range(max_batch):
            self.release_request(slot)
        self.kernel_backend += '+paged-kv'

    def allocate_kv(self):
        return [torch.empty((2, self.pool.capacity, self.kv_heads, 16, self.dim),
                            device=self.device, dtype=self.compute_dtype)
                for _ in range(self.c['num_hidden_layers'])]

    def _sync_pages(self):
        if self.table_revision == self.pool.revision:
            return
        table = torch.full(self.page_table.shape, -1, dtype=torch.int32)
        for slot, pages in enumerate(self.pool.pages):
            if pages:
                table[slot,:len(pages)] = torch.tensor(pages, dtype=torch.int32)
        self.page_table.copy_(table)
        self.table_revision = self.pool.revision

    def reserve_request(self, slot, tokens):
        if self.pool.needed(tokens) > self.pool.capacity:
            raise ValueError('request exceeds the entire KV page budget')
        self.pool.resize_many([(slot,tokens)], replace=True)
        self.reserved_slots.add(slot)
        try:
            self._sync_pages()
        except Exception:
            self.pool.release(slot)
            self.reserved_slots.discard(slot)
            raise

    def release_request(self, slot):
        self.pool.release(slot)
        self.reserved_slots.discard(slot)
        if hasattr(self, 'lengths'):
            self.lengths[slot] = 0
        self._sync_pages()

    def snapshot_prefix(self,slot,count):
        if not 0<count<=self.lengths[slot]:
            raise ValueError('prefix exceeds populated KV cache')
        return [copy_pages(layer,self.page_table,slot,count) for layer in self.kv]

    def restore_prefix(self,slot,count,snapshots):
        if len(snapshots)!=len(self.kv):
            raise ValueError('prefix layer count mismatch')
        size=min(self.context,max(count,len(self.pool.pages[slot])*16)) if slot in self.reserved_slots else count
        self.pool.resize_many([(slot,size)],replace=True)
        self._sync_pages()
        for layer,saved in zip(self.kv,snapshots):
            copy_pages(layer,self.page_table,slot,count,snapshot=saved[:,:,:count].contiguous())

    @property
    def supports_chunked_prefill(self):
        return True

    def _forward(self, ids, positions, slots, prefill_lengths=None, assigned=None, starts=None, return_logits=True):
        w = self.w
        current_kv = prefill_lengths is not None and (starts is None or 0 in starts)
        x = self.embedding(ids)
        for layer in range(self.c['num_hidden_layers']):
            qkv = self.linear(self.norm(x,w[f'{layer}.norm1']), w[f'{layer}.qkv.weight'], w.get(f'{layer}.qkv.bias'))
            ck, cv = self.kv[layer].unbind(0)
            rotated = self.kernels.rope_batch(qkv,self.cos,self.sin,positions,slots,ck,cv,self.heads,
                w.get(f'{layer}.q_norm'),w.get(f'{layer}.k_norm'),self.eps,self.rotary_precise,
                page_table=self.page_table,materialize=current_kv)
            if prefill_lengths is None:
                attended = self.kernels.attention(rotated,ck,cv,positions,slots,
                    window=self.window or 0,page_table=self.page_table).reshape(ids.numel(),-1)
            else:
                query,key,value = rotated if current_kv else (rotated,None,None)
                attended, offset = [], 0
                for slot,length,start in zip(assigned,prefill_lengths,starts or [0]*len(prefill_lengths)):
                    q = query[offset:offset+length,:,0].transpose(0,1).unsqueeze(0)
                    if start:
                        # Offset prefill needs earlier tokens as well as this chunk.
                        dense = copy_pages(self.kv[layer],self.page_table,slot,start+length)
                        k,v = dense[0].unsqueeze(0),dense[1].unsqueeze(0)
                    else:
                        k = key[offset:offset+length].transpose(0,1).unsqueeze(0)
                        v = value[offset:offset+length].transpose(0,1).unsqueeze(0)
                    output = self.prefill_attention(q,k,v,query_start=start)
                    attended.append(output.transpose(1,2).reshape(length,-1))
                    offset += length
                attended = torch.cat(attended)
            x = x + self.linear(attended,w[f'{layer}.out'])
            gate_up = self.linear(self.norm(x,w[f'{layer}.norm2']),w[f'{layer}.gate_up'])
            activated = self.kernels.swiglu(gate_up,self.precise)
            x = x + self.linear(activated,w[f'{layer}.down'])
        if not return_logits:
            return None
        if prefill_lengths is not None:
            ends = torch.tensor([end-1 for end in accumulate(prefill_lengths)],device=self.device)
            x = x.index_select(0,ends)
        return self.output_logits(x)

    @torch.inference_mode()
    def forward(self, ids, positions, prefill, slot=0):
        slots = torch.full_like(positions,slot)
        return self._forward(ids,positions,slots,[ids.numel()] if prefill else None,[slot] if prefill else None)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        return self._forward(ids,positions,slots)

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        return self._prefill_batch(prompts,slots,[0]*len(prompts))

    @torch.inference_mode()
    def prefill_chunk(self,tokens,slot,start,final=True):
        if (type(start) is not int or start<0 or type(slot) is not int or not 0<=slot<self.max_batch
                or (start and self.lengths[slot]!=start)):
            raise ValueError('invalid partial prompt cache state')
        return self._prefill_batch([tokens],[slot],[start],return_logits=final)

    def _prefill_batch(self,prompts,slots,starts,return_logits=True):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(type(slot) is not int or not 0<=slot<self.max_batch for slot in slots)
                or any(not prompt or start+len(prompt)>self.context for prompt,start in zip(prompts,starts))):
            raise ValueError('invalid prefill batch')
        # Admission reservations remain intact; standalone callers replace old slots.
        lengths = [len(prompt) for prompt in prompts]
        sizes = [(slot, min(self.context,max(start+length,len(self.pool.pages[slot])*16)) if slot in self.reserved_slots else start+length)
                 for slot,length,start in zip(slots,lengths,starts)]
        self.pool.resize_many(sizes,replace=True)
        self._sync_pages()
        token_budget = getattr(self, 'prefill_token_budget', self.context)
        if sum(lengths)>token_budget:
            return torch.cat([self.prefill_batch(batch,assigned) for batch,assigned in groups(prompts,slots,token_budget)])
        ids = torch.tensor([token for prompt in prompts for token in prompt],device=self.device)
        positions = torch.tensor([i for length,start in zip(lengths,starts) for i in range(start,start+length)],device=self.device)
        slot_ids = torch.tensor([slot for slot,length in zip(slots,lengths) for _ in range(length)],device=self.device)
        logits = self._forward(ids,positions,slot_ids,lengths,slots,starts,return_logits)
        for slot,length,start in zip(slots,lengths,starts):
            self.lengths[slot] = start+length
        return logits

    def prefill(self, tokens, slot=0):
        logits = self.prefill_batch([tokens],[slot])
        self.length = len(tokens)
        return logits

    def decode_batch(self, tokens, slots):
        if (not tokens or len(tokens)!=len(slots) or len(set(slots))!=len(slots)
                or any(type(slot) is not int or not 0<=slot<self.max_batch for slot in slots)
                or any(not 0<self.lengths[slot]<self.context for slot in slots)):
            raise ValueError('invalid decode batch')
        self.pool.resize_many([(slot,self.lengths[slot]+1) for slot in slots])
        self._sync_pages()
        return super().decode_batch(tokens,slots)

    def decode(self, token):
        result = self.decode_batch([token],[0])
        self.length = self.lengths[0]
        return result
