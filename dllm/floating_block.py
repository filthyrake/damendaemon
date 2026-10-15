"""Prepared floating DeepSeek transformer blocks, independent of serving admission."""
from dataclasses import dataclass
import threading

import torch
from torch.nn import functional as F

from .expert_selection import select_experts, validate_token_expert_table
from .floating_attention import _check, _norm, attention_prefill, attention_decode, AttentionState
from .floating_moe import floating_moe, validate_moe_inputs
from .hyper_connections import reduce_streams, restore_streams
from .packed_moe import PackedExpert, packed_moe
from .prepared_projection import PackedProjection, is_projection


@dataclass
class BlockState:
    owner: object
    attention: AttentionState

    def reorder(self, indices):
        self.attention.reorder(indices)

    def close(self):
        self.attention.close()
        self.owner = None


class FloatingBlock:
    """Prepare resident parameters once; caller must keep their contents immutable.

    Expert banks use stacked [experts,rows,columns] tensors under
    ffn.experts.w1/w3/w2.weight. Other names follow the canonical block layout.
    Immutable hash tables are checked once on CPU during preparation, never by
    extracting routing values in floating prefill/decode. An explicit
    packed_experts/shared pair instead selects the CPU packed expert control;
    that path reads CPU route IDs and supplies no floating expert banks.
    Attention and stream parameters remain prepared floating tensors.
    """
    def __init__(self, attention_weights, weights, frequencies, *, groups, window, ratio=0,
                 hc_mult=4, top_k=6, route_scale=1.5, index_top_k=512, epsilon=1e-6,
                 connection_epsilon=1e-6, iterations=20, activation_limit=10., workspace_bytes=64*1024*1024,
                 packed_experts=None, packed_shared=None, _backend=None):
        if not isinstance(attention_weights, dict) or not isinstance(weights, dict):
            raise ValueError('block preparation requires parameter mappings')
        self.backend = _backend
        projection_check = is_projection if _backend is None else _backend.attention.projection
        packed_type = PackedProjection if _backend is None else _backend.attention.packed_type
        input_weight = attention_weights.get('wq_a.weight')
        if not projection_check(input_weight) or min(input_weight.shape) < 1:
            raise ValueError('missing block input projection geometry')
        probe = (torch.empty(1,1,input_weight.shape[1],dtype=input_weight.dtype,device=input_weight.device)
                 if isinstance(input_weight,packed_type) else input_weight[:1].unsqueeze(0))
        _check(probe, attention_weights, frequencies, groups, window, ratio, index_top_k, 16, epsilon,None if _backend is None else _backend.attention)
        if (type(hc_mult) is not int or not 1 <= hc_mult <= 16
                or type(top_k) is not int or top_k < 1 or type(iterations) is not int or not 1 <= iterations <= 128
                or any(type(v) not in (int, float) or not 0 < v <= torch.finfo(torch.float32).max
                       for v in (route_scale, connection_epsilon))):
            raise ValueError('invalid block routing/hyper-connection options')
        dim, device, dtype = input_weight.shape[1], input_weight.device, input_weight.dtype
        expected = {}
        mix = hc_mult*(hc_mult+2)
        for part in ('attn','ffn'):
            expected[f'hc_{part}_fn'] = (mix,hc_mult*dim)
            expected[f'hc_{part}_base'] = (mix,)
            expected[f'hc_{part}_scale'] = (3,)
            expected[f'{part}_norm.weight'] = (dim,)
        gate = weights.get('ffn.gate.weight')
        if not isinstance(gate, torch.Tensor) or gate.ndim != 2 or gate.shape[0] < top_k:
            raise ValueError('invalid routed gate geometry')
        experts = gate.shape[0]
        expected['ffn.gate.weight'] = (experts,dim)
        table = weights.get('ffn.gate.tid2eid')
        if table is None:
            expected['ffn.gate.bias'] = (experts,)
        for name, shape in expected.items():
            value = weights.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device or value.dtype != torch.float32:
                raise ValueError(f'invalid FP32 block parameter: {name}')
        bank_names = [f'ffn.experts.{p}.weight' for p in ('w1','w3','w2')]
        shared_names = [f'ffn.shared_experts.{p}.weight' for p in ('w1','w3','w2')]
        projection_names = bank_names+shared_names if packed_experts is None else []
        if set(weights) != set(expected) | set(projection_names) | ({'ffn.gate.tid2eid'} if table is not None else set()):
            raise ValueError('block parameter names do not match routing mode')
        if packed_experts is None:
            if packed_shared is not None:
                raise ValueError('packed shared expert requires packed routed experts')
            banks, shared = tuple(weights[n] for n in bank_names), tuple(weights[n] for n in shared_names)
            routes = torch.empty(0,top_k,dtype=torch.int64,device=device)
            coefficients = torch.empty(0,top_k,dtype=torch.float32,device=device)
            validate_moe_inputs(probe.reshape(1,-1)[:0], *banks, routes, coefficients, shared=shared,
                                activation_limit=activation_limit, workspace_bytes=workspace_bytes)
            if banks[0].shape[0] != experts:
                raise ValueError('gate and expert-bank counts disagree')
        else:
            if ((device.type != 'cpu' if _backend is None else device != _backend.device) or not isinstance(packed_experts,(tuple,list)) or len(packed_experts) != experts
                    or type(workspace_bytes) is not int or workspace_bytes < 1):
                raise ValueError('packed expert mode requires a matching CPU expert bank and workspace')
            for expert in (*packed_experts,packed_shared):
                if (not isinstance(expert,PackedExpert if _backend is None else _backend.expert_type) or expert.width != dim
                        or expert.activation_limit != activation_limit or expert.workspace_bytes > workspace_bytes):
                    raise ValueError('packed expert width, activation or workspace differs from block contract')
            packed_experts = tuple(packed_experts)
            banks, shared = None, None
        if table is not None:
            if not isinstance(table, torch.Tensor) or table.ndim != 2 or table.device != device:
                raise ValueError('invalid token-to-expert table geometry/device')
            if _backend is None:
                validate_token_expert_table(table.cpu(), vocab_size=table.shape[0], experts=experts, top_k=top_k)
        self.attention_weights, self.weights, self.frequencies = dict(attention_weights), dict(weights), frequencies
        self.groups, self.window, self.ratio, self.index_top_k = groups, window, ratio, index_top_k
        self.hc_mult, self.top_k, self.route_scale = hc_mult, top_k, route_scale
        self.epsilon, self.connection_epsilon, self.iterations = epsilon, connection_epsilon, iterations
        self.activation_limit, self.workspace_bytes = activation_limit, workspace_bytes
        self.dim, self.device, self.dtype = dim, device, dtype
        self.banks, self.shared, self.table = banks, shared, table
        self.packed_experts, self.packed_shared = packed_experts, packed_shared
        self.failed_prefills=[]
        self.build_lock=threading.RLock()

    def cleanup_failed_prefills(self):
        with self.build_lock:
            while self.failed_prefills:
                self.failed_prefills[0].close()
                self.failed_prefills.pop(0)

    def cache_bytes_per_slot(self,capacity):
        from .floating_attention import cache_bytes_per_slot
        weights=self.attention_weights
        return cache_bytes_per_slot(self.window,self.ratio,capacity,weights['kv_norm.weight'].numel(),
            weights['indexer.compressor.norm.weight'].numel() if self.ratio==4 else None,
            torch.empty((),dtype=self.dtype).element_size())

    def _input(self, hidden, token_ids):
        if self.backend is not None:
            self.backend.admit(hidden,token_ids,self)
            return token_ids
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 4 or min(hidden.shape[:2]) < 1
                or hidden.shape[2:] != (self.hc_mult,self.dim) or hidden.device != self.device or hidden.dtype != self.dtype):
            raise ValueError('block hidden streams do not match prepared parameters')
        if self.table is None:
            return None
        if (not isinstance(token_ids, torch.Tensor) or token_ids.shape != hidden.shape[:2]
                or token_ids.dtype not in (torch.int32,torch.int64) or token_ids.device != self.device):
            raise ValueError('hash-routed blocks require matching integer token IDs')
        # Resolve token bounds before any attention cache can advance.
        return self.table.index_select(0,token_ids.reshape(-1).long())

    def _reduce(self, hidden, part):
        w = self.weights
        reduce = reduce_streams if self.backend is None else self.backend.reduce
        reduced, post, matrix = reduce(hidden, w[f'hc_{part}_fn'], w[f'hc_{part}_scale'], w[f'hc_{part}_base'],
                                               norm_epsilon=self.epsilon, connection_epsilon=self.connection_epsilon,
                                               iterations=self.iterations)
        return _norm(reduced,w[f'{part}_norm.weight'],self.epsilon), post, matrix

    def _ffn(self, hidden, token_routes):
        reduced, post, matrix = self._reduce(hidden,'ffn')
        if self.backend is not None:
            output=self.backend.ffn(reduced,token_routes)
            return self.backend.restore(output,hidden,post,matrix)
        rows = reduced.flatten(0,1)
        logits = F.linear(rows.float(),self.weights['ffn.gate.weight'])
        coefficients, selected = select_experts(logits,self.top_k,scoring='sqrtsoftplus',renormalize=True,
                                                scale=self.route_scale,selected_experts=token_routes,
                                                correction_bias=self.weights.get('ffn.gate.bias'))
        if self.packed_experts is None:
            output = floating_moe(rows,*self.banks,selected.contiguous(),coefficients.contiguous(),shared=self.shared,
                                  activation_limit=self.activation_limit,workspace_bytes=self.workspace_bytes)
        else:
            output = packed_moe(rows,self.packed_experts,selected,coefficients,shared=self.packed_shared)
        return restore_streams(output.reshape(reduced.shape),hidden,post,matrix)

    @torch.no_grad()
    def prefill(self, hidden, token_ids=None, *, max_tokens=None, max_cache_bytes=256*1024*1024, query_chunk=16):
        self.cleanup_failed_prefills()
        routes = self._input(hidden,token_ids)
        reduced, post, matrix = self._reduce(hidden,'attn')
        prefill = attention_prefill if self.backend is None else self.backend.prefill
        restore = restore_streams if self.backend is None else self.backend.restore
        output, attention = prefill(reduced,self.attention_weights,self.frequencies,groups=self.groups,
                                              window=self.window,ratio=self.ratio,index_top_k=self.index_top_k,
                                              epsilon=self.epsilon,max_tokens=max_tokens,max_cache_bytes=max_cache_bytes,
                                              query_chunk=query_chunk)
        try:
            result = self._ffn(restore(output,hidden,post,matrix),routes)
        except BaseException as error:
            try:attention.close()
            except BaseException:
                with self.build_lock:self.failed_prefills.append(attention)
                error.add_note('failed block prefill cleanup remains owned; retry before another prefill')
            raise
        return result, BlockState(self,attention)

    @torch.no_grad()
    def decode(self, hidden, state, token_ids=None, *, position):
        if not isinstance(state,BlockState) or state.owner is not self:
            raise ValueError('block cache belongs to a different or closed owner')
        with state.attention.lock:
            state.attention._usable()
            if type(position) is not int or position != state.attention.position or position >= state.attention.capacity:
                raise ValueError('stale block position or capacity exhausted')
            routes = self._input(hidden,token_ids)
            if hidden.shape[1] != 1:
                raise ValueError('block decode requires exactly one token per slot')
            reduced, post, matrix = self._reduce(hidden,'attn')
            restore = restore_streams if self.backend is None else self.backend.restore
            advanced = False
            try:
                output = attention_decode(reduced,state.attention,position=position)
                advanced = True
                return self._ffn(restore(output,hidden,post,matrix),routes)
            except BaseException:
                if advanced:
                    state.attention.poisoned = True
                raise
