"""Bounded executor-owned KV snapshots, copied into independent request slots."""
from collections import OrderedDict
import torch
import weakref
from .device import available_memory,empty_cache
from .request_scope import CacheScope


class PrefixCache:
    def __init__(self,executor,token_budget,max_entries=4):
        if type(token_budget) is not int or token_budget<=0 or max_entries<=0:
            raise ValueError('invalid prefix cache budget')
        self.executor=weakref.proxy(executor)
        self.token_budget=token_budget
        self.max_entries=max_entries
        self.entries=OrderedDict()
        self.tokens=0

    @torch.inference_mode()
    def restore(self,prompt,slot,*,scope=None):
        best=None
        reused=0
        for key,(cache,logits) in self.entries.items():
            entry_scope, tokens = key
            if entry_scope is not scope:continue
            common=0
            for left,right in zip(tokens,prompt):
                if left!=right:break
                common+=1
            exact=common==len(prompt)==len(tokens) and logits is not None
            if not getattr(self.executor,'prefix_can_rewind',True) and common!=len(tokens):
                continue
            count=common if exact else min(common,len(prompt)-1)
            if not getattr(self.executor,'prefix_can_rewind',True) and count!=len(tokens):continue
            if count>reused:
                best=(key,cache,logits if exact else None)
                reused=count
        if best is None:return 0,None
        key,cache,logits=best
        if hasattr(self.executor,'restore_prefix'):
            self.executor.restore_prefix(slot,reused,cache)
        else:
            for layer,saved in zip(self.executor.kv,cache):
                layer[:,slot,:,:reused,:].copy_(saved[:,:,:reused,:])
        self.executor.lengths[slot]=reused
        self.entries.move_to_end(key)
        return reused,None if logits is None else logits.clone()

    @torch.inference_mode()
    def remember(self,prompt,slot,logits,*,scope=None):
        scope_limit = scope.max_prefix_cache_tokens if isinstance(scope, CacheScope) else None
        budget = self.token_budget if scope_limit is None else min(self.token_budget, scope_limit)
        if not budget or (not getattr(self.executor,'prefix_can_rewind',True) and len(prompt)>budget):
            return
        count=min(len(prompt),budget)
        key=(scope,tuple(prompt[:count]))
        if not count or key in self.entries:return
        if scope_limit is not None:
            owned = [key for key in self.entries if key[0] is scope]
            retained = sum(len(key[1]) for key in owned)
            for old in owned:
                if retained+count <= scope_limit:break
                del self.entries[old]
                retained -= len(old[1])
                self.tokens -= len(old[1])
        while self.entries and (self.tokens+count>self.token_budget or len(self.entries)>=self.max_entries):
            old,_=self.entries.popitem(last=False)
            self.tokens-=len(old[1])
        if hasattr(self.executor,'prefix_state_bytes'):
            needed=self.executor.prefix_state_bytes(count)
        else:
            needed=sum(2*layer.shape[2]*count*layer.shape[-1]*layer.element_size() for layer in self.executor.kv)
        if count==len(prompt):needed+=logits.numel()*logits.element_size()
        if needed+256*1024**2>available_memory():
            empty_cache()
            if needed+256*1024**2>available_memory():return
        snapshots=[]
        try:
            if hasattr(self.executor,'snapshot_prefix'):
                snapshots=self.executor.snapshot_prefix(slot,count)
            else:
                for layer in self.executor.kv:
                    snapshots.append(layer[:,slot,:,:count,:].clone())
            saved_logits=logits.clone() if count==len(prompt) else None
        except torch.OutOfMemoryError:
            snapshots.clear()
            empty_cache()
            return
        self.entries[key]=(snapshots,saved_logits)
        self.tokens+=count
